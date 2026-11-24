import assert from 'node:assert/strict';
import test, {type TestContext} from 'node:test';

import {
  ForecastWorkerPool,
  runIsolatedForecast,
} from '../../src/components/ModelingLabClient/forecastWorker';

type Request = {id: number; wasmJsUrl: string; wasmBinaryUrl: string; request: unknown};

function installWorkerHarness(t: TestContext) {
  const originalWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  const workers: FakeWorker[] = [];
  const created: string[] = [];
  const revoked: string[] = [];
  let constructorError: Error | undefined;
  let postError: Error | undefined;

  class FakeWorker {
    onmessage: ((event: {data: unknown}) => void) | null = null;
    onerror: ((event: {message: string}) => void) | null = null;
    onmessageerror: (() => void) | null = null;
    messages: Request[] = [];
    terminations = 0;

    constructor(readonly url: string, readonly options: {type: string}) {
      if (constructorError) throw constructorError;
      workers.push(this);
    }

    postMessage(message: Request) {
      if (postError) throw postError;
      this.messages.push(message);
    }

    terminate() {
      this.terminations += 1;
    }

    reply(id: number, response: unknown) {
      this.onmessage?.({data: {id, response}});
    }
  }

  Object.defineProperty(globalThis, 'Worker', {configurable: true, value: FakeWorker});
  t.mock.method(URL, 'createObjectURL', () => {
    const url = `blob:forecast-${created.length + 1}`;
    created.push(url);
    return url;
  });
  t.mock.method(URL, 'revokeObjectURL', (url: string) => revoked.push(url));
  t.after(() => {
    if (originalWorker) Object.defineProperty(globalThis, 'Worker', originalWorker);
    else Reflect.deleteProperty(globalThis, 'Worker');
  });
  return {
    workers,
    created,
    revoked,
    failConstruction(error?: Error) {constructorError = error;},
    failPost(error?: Error) {postError = error;},
  };
}

test('forecast worker reuses its session and routes concurrent replies by request id', async (t) => {
  const harness = installWorkerHarness(t);
  const pool = new ForecastWorkerPool<{forecast: number}>();
  t.after(() => pool.dispose());
  const first = pool.run('https://example.test/model.js', 'https://example.test/model.wasm', {horizon: 1});
  const second = pool.run('https://example.test/model.js', 'https://example.test/model.wasm', {horizon: 2});
  assert.equal(harness.workers.length, 1);
  const worker = harness.workers[0];
  assert.deepEqual(worker.options, {type: 'module'});
  assert.deepEqual(worker.messages, [
    {id: 1, wasmJsUrl: 'https://example.test/model.js', wasmBinaryUrl: 'https://example.test/model.wasm', request: {horizon: 1}},
    {id: 2, wasmJsUrl: 'https://example.test/model.js', wasmBinaryUrl: 'https://example.test/model.wasm', request: {horizon: 2}},
  ]);
  worker.reply(2, {forecast: 20});
  worker.reply(1, {forecast: 10});
  assert.deepEqual(await Promise.all([first, second]), [{forecast: 10}, {forecast: 20}]);
  pool.dispose();
  pool.dispose();
  assert.equal(worker.terminations, 1);
  assert.deepEqual(harness.revoked, harness.created);
});

test('changing the bundle rejects pending requests before replacing the worker', async (t) => {
  const harness = installWorkerHarness(t);
  const pool = new ForecastWorkerPool<object>();
  t.after(() => pool.dispose());
  const oldRequest = pool.run('old.js', 'old.wasm', {});
  const rejected = assert.rejects(oldRequest, /replaced by a different model bundle/);
  const newRequest = pool.run('new.js', 'new.wasm', {});
  await rejected;
  assert.equal(harness.workers[0].terminations, 1);
  assert.deepEqual(harness.revoked, [harness.created[0]]);
  harness.workers[1].reply(1, {forecast: 7});
  assert.deepEqual(await newRequest, {forecast: 7});
});

for (const event of ['error', 'messageerror'] as const) {
  test(`worker ${event} rejects all pending requests and allows a fresh session`, async (t) => {
    const harness = installWorkerHarness(t);
    const pool = new ForecastWorkerPool<object>();
    t.after(() => pool.dispose());
    const first = pool.run('model.js', 'model.wasm', {});
    const second = pool.run('model.js', 'model.wasm', {});
    const message = event === 'error' ? /worker crashed/ : /unreadable response/;
    const failures = Promise.all([assert.rejects(first, message), assert.rejects(second, message)]);
    const worker = harness.workers[0];
    if (event === 'error') worker.onerror?.({message: 'worker crashed'});
    else worker.onmessageerror?.();
    await failures;
    assert.equal(worker.terminations, 1);
    assert.deepEqual(harness.revoked, [harness.created[0]]);
    const retry = pool.run('model.js', 'model.wasm', {});
    assert.equal(harness.workers.length, 2);
    harness.workers[1].reply(1, {forecast: 3});
    assert.deepEqual(await retry, {forecast: 3});
  });
}

test('worker creation failure revokes its URL and permits a later retry', async (t) => {
  const harness = installWorkerHarness(t);
  const pool = new ForecastWorkerPool<object>();
  t.after(() => pool.dispose());
  harness.failConstruction(new Error('worker unavailable'));
  await assert.rejects(pool.run('model.js', 'model.wasm', {}), /worker unavailable/);
  assert.deepEqual(harness.revoked, harness.created);
  harness.failConstruction();
  const retry = pool.run('model.js', 'model.wasm', {});
  harness.workers[0].reply(1, {forecast: 3});
  assert.deepEqual(await retry, {forecast: 3});
});

test('postMessage failure rejects only its request without poisoning the session', async (t) => {
  const harness = installWorkerHarness(t);
  const pool = new ForecastWorkerPool<object>();
  t.after(() => pool.dispose());
  const first = pool.run('model.js', 'model.wasm', {});
  harness.failPost(new Error('request cannot be cloned'));
  await assert.rejects(pool.run('model.js', 'model.wasm', {}), /cannot be cloned/);
  harness.failPost();
  const third = pool.run('model.js', 'model.wasm', {});
  assert.equal(harness.workers.length, 1);
  harness.workers[0].reply(1, {forecast: 1});
  harness.workers[0].reply(3, {forecast: 3});
  assert.deepEqual(await Promise.all([first, third]), [{forecast: 1}, {forecast: 3}]);
});

test('isolated workers terminate and revoke their URL after success or a model error', async (t) => {
  const harness = installWorkerHarness(t);
  const success = runIsolatedForecast('model.js', 'model.wasm', {}, 10_000, 'timed out');
  harness.workers[0].reply(1, {forecast: 5});
  assert.deepEqual(await success, {forecast: 5});
  const failure = runIsolatedForecast('model.js', 'model.wasm', {}, 10_000, 'timed out');
  const rejected = assert.rejects(failure, /invalid model/);
  harness.workers[1].onmessage?.({data: {id: 1, error: 'invalid model'}});
  await rejected;
  assert.deepEqual(harness.workers.map((worker) => worker.terminations), [1, 1]);
  assert.deepEqual(harness.revoked, harness.created);
});

test('isolated timeout rejects the request and releases worker resources', async (t) => {
  const harness = installWorkerHarness(t);
  await assert.rejects(
    runIsolatedForecast('model.js', 'model.wasm', {}, 1, 'forecast timed out'),
    /forecast timed out/,
  );
  assert.equal(harness.workers[0].terminations, 1);
  assert.deepEqual(harness.revoked, harness.created);
});

test('isolated postMessage failure releases worker resources immediately', async (t) => {
  const harness = installWorkerHarness(t);
  harness.failPost(new Error('request cannot be cloned'));
  await assert.rejects(
    runIsolatedForecast('model.js', 'model.wasm', {}, 10_000, 'timed out'),
    /cannot be cloned/,
  );
  assert.equal(harness.workers[0].terminations, 1);
  assert.deepEqual(harness.revoked, harness.created);
});
