/**
 * Tiny toast queue for optimistic-UI failures.
 */

import type { ToastMessage } from '$lib/types';

/**
 * `crypto.randomUUID` only exists in "secure contexts" (HTTPS or
 * `localhost`) per the Web Crypto spec. Safari enforces this strictly;
 * Chrome is more lenient. This app is routinely accessed over plain HTTP
 * via a LAN IP (not localhost, not HTTPS), where Safari has no
 * `crypto.randomUUID` at all — calling it throws a TypeError. Toast ids
 * only need to be unique within this tab's session, not cryptographically
 * random, so a plain fallback avoids the secure-context restriction
 * entirely instead of trying to work around it.
 */
function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

class ToastStore {
  toasts = $state<ToastMessage[]>([]);

  push(t: Omit<ToastMessage, 'id'>): string {
    const id = makeId();
    const ttl = t.ttl_ms ?? 4500;
    const msg: ToastMessage = { id, ttl_ms: ttl, ...t };
    this.toasts = [...this.toasts, msg];
    if (ttl > 0) {
      setTimeout(() => this.dismiss(id), ttl);
    }
    return id;
  }

  dismiss(id: string): void {
    this.toasts = this.toasts.filter((t) => t.id !== id);
  }

  error(text: string): string {
    return this.push({ kind: 'error', text });
  }

  success(text: string): string {
    return this.push({ kind: 'success', text });
  }

  info(text: string): string {
    return this.push({ kind: 'info', text });
  }

  warn(text: string): string {
    return this.push({ kind: 'warn', text });
  }
}

export const toastStore = new ToastStore();
