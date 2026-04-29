/**
 * Tiny toast queue for optimistic-UI failures.
 */

import type { ToastMessage } from '$lib/types';

class ToastStore {
  toasts = $state<ToastMessage[]>([]);

  push(t: Omit<ToastMessage, 'id'>): string {
    const id = crypto.randomUUID();
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
