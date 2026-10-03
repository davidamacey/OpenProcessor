import type { HandleClientError } from '@sveltejs/kit';
import {
  errorMessage,
  isChunkLoadError,
  recoverFromChunkError,
} from '$lib/chunkRecovery';

function sessionStore(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function recover(): boolean {
  return recoverFromChunkError(sessionStore(), Date.now(), () =>
    window.location.reload(),
  );
}

// A preload failure raised by Vite's own helper; the default is to throw it
// into whoever awaited the import, so recover here as well.
window.addEventListener('vite:preloadError', (event) => {
  if (recover()) event.preventDefault();
});

export const handleError: HandleClientError = ({ error }) => {
  console.error(error);
  if (isChunkLoadError(error)) {
    // The thrown message is kept (SvelteKit would otherwise reduce it to a
    // bare "Internal Error") so +error.svelte can show the cause.
    recover();
    return { message: errorMessage(error) };
  }
};
