/** A reactive stand-in for SvelteKit's `page` from `$app/state`, for tests. */
import { SvelteURL } from 'svelte/reactivity';

export const page = $state({ url: new SvelteURL('http://localhost/') });

export function visit(href: string): void {
  page.url = new SvelteURL(href, 'http://localhost');
}
