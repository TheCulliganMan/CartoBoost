import React, { useState } from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

type CopyState = 'idle' | 'copied' | 'error';

async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', 'true');
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand('copy');
  document.body.removeChild(textarea);
  if (!copied) {
    throw new Error('copy failed');
  }
}

export default function AgentLlmsCallout(): React.JSX.Element {
  const llmsUrl = useBaseUrl('/docs/llms.txt');
  const [state, setState] = useState<CopyState>('idle');

  async function copyLlms(): Promise<void> {
    try {
      const response = await fetch(llmsUrl, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`failed to fetch ${llmsUrl}`);
      }
      await copyToClipboard(await response.text());
      setState('copied');
    } catch {
      setState('error');
    }
  }

  const status =
    state === 'copied' ? 'Copied' : state === 'error' ? 'Copy failed' : 'Ready';

  return (
    <div className="agent-llms-callout">
      <div>
        <h2>LLM Guide</h2>
        <p>
          Give your coding assistant a map of the package, public APIs, and
          documentation with the plain-text llms.txt guide.
        </p>
      </div>
      <div className="agent-llms-actions">
        <button className="agent-copy-button agent-copy-button--primary" type="button" onClick={copyLlms}>
          Copy llms.txt
        </button>
        <a className="agent-copy-link" href={llmsUrl}>
          Open llms.txt
        </a>
        <a className="agent-copy-link" href={llmsUrl} download>
          Download
        </a>
        <span className="agent-copy-status" aria-live="polite">
          {status}
        </span>
      </div>
    </div>
  );
}
