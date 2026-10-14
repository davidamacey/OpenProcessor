/**
 * `packsAvailability`: the "not yet deployed" gate for prompt packs
 * (CRUD). The spec gives them no capability signal, so this
 * probes `GET {prefix}/prompt_packs` once per project (404/501 = every
 * pack surface absent). See `ConfigAvailability`. Reset on
 * a project switch.
 */
import { listPromptPacks } from '$lib/api_packs';
import { ConfigAvailability } from '$lib/config/configAvailability.svelte';
import { onProjectChange } from '$lib/projectChange';

export const packsAvailability = new ConfigAvailability(() => listPromptPacks());

onProjectChange(() => packsAvailability.reset());
