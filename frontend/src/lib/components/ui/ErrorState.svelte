<script lang="ts">
  /**
   * A failed read: the served message, the request id (so a report can be traced in the
   * API log) and a Retry. Pass an `ApiError` as `error`, or a `message` and `requestId`.
   */
  import { ApiError } from '$lib/api_errors';
  import Button from './Button.svelte';
  import Icon from './Icon.svelte';

  interface Props {
    title?: string;
    message?: string;
    requestId?: string | null;
    error?: unknown;
    onretry?: () => void;
  }
  let {
    title = 'Something went wrong',
    message,
    requestId,
    error,
    onretry,
  }: Props = $props();

  const shownMessage = $derived(
    message ??
      (error instanceof ApiError
        ? (error.detail ?? `The server answered ${error.status}`)
        : error instanceof Error
          ? error.message
          : ''),
  );
  const shownId = $derived(
    requestId ?? (error instanceof ApiError ? error.requestId : null),
  );
</script>

<div
  class="flex flex-col items-start gap-2 rounded-card border border-danger-line bg-danger-soft p-4"
  role="alert"
  data-testid="error-state"
>
  <div class="flex items-center gap-2 text-title font-semibold text-danger-text">
    <Icon name="triangle-alert" size={18} />
    {title}
  </div>
  {#if shownMessage}
    <p class="text-body text-fg-1" data-testid="error-state-message">{shownMessage}</p>
  {/if}
  {#if shownId}
    <p class="text-meta text-fg-2">
      Request id <code class="text-fg-1" data-testid="error-state-request-id"
        >{shownId}</code
      >
    </p>
  {/if}
  {#if onretry}
    <Button size="sm" icon="refresh-cw" onclick={onretry}>Retry</Button>
  {/if}
</div>
