/**
 * `openVocabAvailability`: the served-feature gate for OpenProcessor v0.4.0's
 * SAM 3 open-vocabulary sets. No capability flag is served, so this probes
 * `GET {prefix}/open_vocab` once per project (404/501 = every open-vocab
 * surface absent, nothing else fires). See `ConfigAvailability`. Reset on a
 * project switch.
 */
import { listOpenVocab } from '$lib/api_openVocab';
import { ConfigAvailability } from '$lib/config/configAvailability.svelte';
import { onProjectChange } from '$lib/projectChange';

export const openVocabAvailability = new ConfigAvailability(() => listOpenVocab());

onProjectChange(() => openVocabAvailability.reset());
