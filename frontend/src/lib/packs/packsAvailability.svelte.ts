/**
 * `packsAvailability`: the "not yet deployed" gate for OpenProcessor W3
 * (prompt-pack CRUD). The spec gives W3 no capability signal, so this
 * probes `GET {prefix}/prompt_packs` once per project (404/501 = every
 * pack surface absent). See `ConfigAvailability` and
 * docs/design/w3-pack-editor-ui-plan-2026-09-27.md §0.3 (W3-Q1). Reset on
 * a project switch.
 */
import { listPromptPacks } from '$lib/api';
import { ConfigAvailability } from '$lib/config/configAvailability.svelte';
import { onProjectChange } from '$lib/projectChange';

export const packsAvailability = new ConfigAvailability(() => listPromptPacks());

onProjectChange(() => packsAvailability.reset());
