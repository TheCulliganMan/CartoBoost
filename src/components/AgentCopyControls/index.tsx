import React, { useState } from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

type CopyState = 'idle' | 'page' | 'llms' | 'error';

function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
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
  return copied ? Promise.resolve() : Promise.reject(new Error('copy failed'));
}

function pageText(): string {
  const article = document.querySelector('article');
  return article?.textContent?.trim() ?? document.body.textContent?.trim() ?? '';
}

export default function AgentCopyControls(): React.JSX.Element {
  const llmsUrl = useBaseUrl('/docs/llms.txt');
  const [state, setState] = useState<CopyState>('idle');

  async function copyPage(): Promise<void> {
    try {
      await copyText(pageText());
      setState('page');
    } catch {
      setState('error');
    }
  }

  async function copyLlms(): Promise<void> {
    try {
      const response = await fetch(llmsUrl, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`failed to fetch ${llmsUrl}`);
      }
      await copyText(await response.text());
      setState('llms');
    } catch {
      setState('error');
    }
  }

  const status =
    state === 'page'
      ? 'Page copied'
      : state === 'llms'
        ? 'LLM guide copied'
        : state === 'error'
          ? 'Copy failed'
          : '';

  return (
    <div className="agent-copy-controls" aria-label="Copy tools">
      <button className="agent-copy-button" type="button" onClick={copyPage}>
        Copy Page
      </button>
      <button className="agent-copy-button" type="button" onClick={copyLlms}>
        Copy llms.txt
      </button>
      <a className="agent-copy-link" href={llmsUrl}>
        Open llms.txt
      </a>
      <span className="agent-copy-status" aria-live="polite">
        {status}
      </span>
    </div>
  );
}
