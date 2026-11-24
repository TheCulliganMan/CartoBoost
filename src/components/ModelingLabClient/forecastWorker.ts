type ForecastWorkerReply<Response> = {id: number; response?: Response; error?: string};
type Pending<Response> = {resolve: (value: Response) => void; reject: (error: Error) => void};

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

class ForecastWorkerSession<Response> {
  private readonly worker: Worker;
  private readonly blobUrl: string;
  private readonly pending = new Map<number, Pending<Response>>();
  private nextId = 1;
  closed = false;

  constructor(private readonly wasmJsUrl: string, private readonly wasmBinaryUrl: string) {
    this.blobUrl = URL.createObjectURL(new Blob([forecastWorkerSource], {type: 'text/javascript'}));
    try {
      this.worker = new Worker(this.blobUrl, {type: 'module'});
    } catch (error) {
      URL.revokeObjectURL(this.blobUrl);
      throw error;
    }
    this.worker.onmessage = (event: MessageEvent<ForecastWorkerReply<Response>>) => {
      const {id, response, error} = event.data;
      const callbacks = this.pending.get(id);
      if (!callbacks) {
        return;
      }
      this.pending.delete(id);
      if (error) {
        callbacks.reject(new Error(error));
      } else if (!response) {
        callbacks.reject(new Error('forecast worker returned an empty response'));
      } else {
        callbacks.resolve(response);
      }
    };
    this.worker.onerror = (event) => this.dispose(new Error(event.message || 'forecast worker failed'));
    this.worker.onmessageerror = () => this.dispose(new Error('forecast worker returned an unreadable response'));
  }

  run(request: unknown): Promise<Response> {
    if (this.closed) {
      return Promise.reject(new Error('forecast worker has stopped'));
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      try {
        this.worker.postMessage({id, wasmJsUrl: this.wasmJsUrl, wasmBinaryUrl: this.wasmBinaryUrl, request});
      } catch (error) {
        this.pending.delete(id);
        reject(asError(error));
      }
    });
  }

  dispose(error = new Error('forecast worker has stopped')) {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.worker.onmessage = null;
    this.worker.onerror = null;
    this.worker.onmessageerror = null;
    this.worker.terminate();
    URL.revokeObjectURL(this.blobUrl);
    for (const callbacks of this.pending.values()) {
      callbacks.reject(error);
    }
    this.pending.clear();
  }
}

export class ForecastWorkerPool<Response> {
  private key = '';
  private client: ForecastWorkerSession<Response> | null = null;

  async run(wasmJsUrl: string, wasmBinaryUrl: string, request: unknown): Promise<Response> {
    const key = `${wasmJsUrl}\u0000${wasmBinaryUrl}`;
    if (!this.client || this.client.closed || this.key !== key) {
      this.client?.dispose(new Error('forecast worker replaced by a different model bundle'));
      this.client = null;
      this.client = new ForecastWorkerSession<Response>(wasmJsUrl, wasmBinaryUrl);
      this.key = key;
    }
    return this.client.run(request);
  }

  dispose() {
    this.client?.dispose();
    this.client = null;
    this.key = '';
  }
}

export async function runIsolatedForecast<Response>(
  wasmJsUrl: string,
  wasmBinaryUrl: string,
  request: unknown,
  timeoutMs: number,
  timeoutMessage: string,
): Promise<Response> {
  const client = new ForecastWorkerSession<Response>(wasmJsUrl, wasmBinaryUrl);
  const timeout = setTimeout(() => client.dispose(new Error(timeoutMessage)), timeoutMs);
  try {
    return await client.run(request);
  } finally {
    clearTimeout(timeout);
    client.dispose();
  }
}

const forecastWorkerSource = `
let wasmKey = '';
let wasmPromise = null;

async function getWasmModule(wasmJsUrl, wasmBinaryUrl) {
  const key = wasmJsUrl + '\\0' + wasmBinaryUrl;
  if (wasmPromise && wasmKey === key) {
    return wasmPromise;
  }
  wasmKey = key;
  wasmPromise = (async () => {
    const response = await fetch(wasmJsUrl, {cache: 'no-store'});
    const contentType = response.headers.get('content-type') || '';
    if (!response.ok || contentType.includes('text/html')) {
      throw new Error('CartoBoost interactive model bundle is not available from this dev server.');
    }
    const blobUrl = URL.createObjectURL(
      new Blob([await response.text()], {type: contentType.includes('javascript') ? contentType : 'text/javascript'}),
    );
    try {
      const wasmModule = await import(blobUrl);
      await wasmModule.default({module_or_path: wasmBinaryUrl});
      return wasmModule;
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
  })();
  const pending = wasmPromise;
  try {
    return await pending;
  } catch (error) {
    if (wasmPromise === pending) {
      wasmPromise = null;
      wasmKey = '';
    }
    throw error;
  }
}

self.onmessage = async (event) => {
  const {id, wasmJsUrl, wasmBinaryUrl, request} = event.data;
  try {
    const wasmModule = await getWasmModule(wasmJsUrl, wasmBinaryUrl);
    self.postMessage({id, response: wasmModule.runForecast(request)});
  } catch (error) {
    self.postMessage({id, error: error instanceof Error ? error.message : String(error)});
  }
};
`;
