<script lang="ts">
  /** The app mark. With a `label` it is an image for assistive tech; without, decoration. */
  interface Props {
    class?: string;
    label?: string;
  }

  let { class: className = '', label }: Props = $props();
</script>

<svg
  viewBox="0 0 128 128"
  class={className}
  role={label ? 'img' : undefined}
  aria-label={label}
>
  <rect width="128" height="128" rx="24" fill="#09090b" />
  <rect x="26" y="70" width="30" height="30" rx="5" fill="#3f3f46" />
  <rect x="60" y="70" width="30" height="30" rx="5" fill="#3f3f46" />
  <rect x="26" y="34" width="30" height="30" rx="5" fill="#60a5fa" />
  <rect x="60" y="34" width="30" height="30" rx="5" fill="#f59e0b" />
  <path
    d="M33 49 l6 6 l12 -12"
    fill="none"
    stroke="#09090b"
    stroke-width="4"
    stroke-linecap="round"
    stroke-linejoin="round"
  />
</svg>
