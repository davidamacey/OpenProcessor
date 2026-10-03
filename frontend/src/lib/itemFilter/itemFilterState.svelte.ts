/**
 * The operator's shared item-filter choices (OpenProcessor v0.4.0): one
 * state object read by /review, /clusters (grid and "Matching items") and
 * /export. It only holds what the operator picked and spells it in the two
 * wire forms (query parameters for list/stats routes, the `ItemFilter` body
 * for `selection` / `export/yolo item_filter`); the server decides what a
 * filter matches, how many items that is and which values are valid.
 */
import { humanizeId } from '$lib/humanizeId';
import {
  EMBEDDING_STATES,
  ITEM_ORIGINS,
  REVIEW_STATUSES,
  type EmbeddingState,
  type ItemFilter,
  type ItemFilterQuery,
  type ItemOrigin,
  type ReviewStatus,
} from '$lib/types_itemFilter';

/** `GET /clusters`, `/regions`, `/search/text`, `/review/{tab}` and
 *  `/stats/dataset` do not declare `open_vocab_set` / `source_prompt`
 *  (`GET /crops` alone does). */
export const withoutOpenVocab = (param: string): boolean =>
  param !== 'open_vocab_set' && param !== 'source_prompt';

export interface ItemFilterChip {
  param: string;
  label: string;
  clear(): void;
}

function pick<T extends string>(values: string[], allowed: readonly T[]): T[] {
  return values.filter((v): v is T => (allowed as readonly string[]).includes(v));
}

export class ItemFilterState {
  classNames = $state<string[]>([]);
  excludeClassNames = $state<string[]>([]);
  confMin = $state<number | null>(null);
  confMax = $state<number | null>(null);
  minArea = $state<number | null>(null);
  maxArea = $state<number | null>(null);
  maxRank = $state<number | null>(null);
  origin = $state<ItemOrigin[]>([]);
  embeddingState = $state<EmbeddingState[]>([]);
  reviewStatus = $state<ReviewStatus[]>([]);
  openVocabSet = $state<string | null>(null);
  sourcePrompt = $state<string | null>(null);

  get isEmpty(): boolean {
    return (
      this.classNames.length === 0 &&
      this.excludeClassNames.length === 0 &&
      this.confMin == null &&
      this.confMax == null &&
      this.minArea == null &&
      this.maxArea == null &&
      this.maxRank == null &&
      this.origin.length === 0 &&
      this.embeddingState.length === 0 &&
      this.reviewStatus.length === 0 &&
      !this.openVocabSet &&
      !this.sourcePrompt
    );
  }

  /** Query-parameter form. `allowed` lets a route (or a review tab's served
   *  `filters`) drop params it does not honour. Empty values are omitted. */
  toQuery(allowed: (param: string) => boolean = () => true): ItemFilterQuery {
    const q: Record<string, unknown> = {};
    const put = (param: string, value: unknown, present: boolean): void => {
      if (present && allowed(param)) q[param] = value;
    };
    put('class_name', [...this.classNames], this.classNames.length > 0);
    put(
      'exclude_class_name',
      [...this.excludeClassNames],
      this.excludeClassNames.length > 0,
    );
    put('conf_min', this.confMin, this.confMin != null);
    put('conf_max', this.confMax, this.confMax != null);
    put('min_area', this.minArea, this.minArea != null);
    put('max_area', this.maxArea, this.maxArea != null);
    put('max_rank', this.maxRank, this.maxRank != null);
    put('origin', [...this.origin], this.origin.length > 0);
    put('embedding_state', [...this.embeddingState], this.embeddingState.length > 0);
    put('review_status', [...this.reviewStatus], this.reviewStatus.length > 0);
    put('open_vocab_set', this.openVocabSet, !!this.openVocabSet);
    put('source_prompt', this.sourcePrompt, !!this.sourcePrompt);
    return q as ItemFilterQuery;
  }

  /** The `ItemFilter` request-body form (`selection.filter`, `item_filter`). */
  toBody(): ItemFilter {
    const b: ItemFilter = {};
    if (this.classNames.length > 0) b.class_names = [...this.classNames];
    if (this.excludeClassNames.length > 0) {
      b.exclude_class_names = [...this.excludeClassNames];
    }
    if (this.confMin != null) b.conf_min = this.confMin;
    if (this.confMax != null) b.conf_max = this.confMax;
    if (this.minArea != null) b.min_area = this.minArea;
    if (this.maxArea != null) b.max_area = this.maxArea;
    if (this.maxRank != null) b.max_rank = this.maxRank;
    if (this.origin.length > 0) b.origin = [...this.origin];
    if (this.embeddingState.length > 0) b.embedding_state = [...this.embeddingState];
    if (this.reviewStatus.length > 0) b.review_status = [...this.reviewStatus];
    if (this.openVocabSet) b.open_vocab_set = this.openVocabSet;
    if (this.sourcePrompt) b.source_prompt = this.sourcePrompt;
    return b;
  }

  /** Reads the same names `toUrl` writes (repeatable keys). */
  fromUrl(params: URLSearchParams): void {
    const num = (k: string): number | null => {
      const v = params.get(k);
      if (v == null || v === '') return null;
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    };
    this.classNames = params.getAll('class_name');
    this.excludeClassNames = params.getAll('exclude_class_name');
    this.confMin = num('conf_min');
    this.confMax = num('conf_max');
    this.minArea = num('min_area');
    this.maxArea = num('max_area');
    this.maxRank = num('max_rank');
    this.origin = pick(params.getAll('origin'), ITEM_ORIGINS);
    this.embeddingState = pick(params.getAll('embedding_state'), EMBEDDING_STATES);
    this.reviewStatus = pick(params.getAll('review_status'), REVIEW_STATUSES);
    this.openVocabSet = params.get('open_vocab_set') || null;
    this.sourcePrompt = params.get('source_prompt') || null;
  }

  /** Writes the state into `params`, deleting the keys that are now empty. */
  toUrl(params: URLSearchParams): void {
    const query = this.toQuery() as Record<string, unknown>;
    for (const key of URL_KEYS) {
      params.delete(key);
      const v = query[key];
      if (Array.isArray(v)) for (const x of v) params.append(key, String(x));
      else if (v != null) params.set(key, String(v));
    }
  }

  /** The control value of a param by its query name: a list for the
   *  list-valued ones, otherwise a string (`''` = unset). */
  valueOf(param: string): string | string[] | undefined {
    const num = (n: number | null): string => (n == null ? '' : String(n));
    switch (param) {
      case 'class_name':
        return this.classNames;
      case 'exclude_class_name':
        return this.excludeClassNames;
      case 'conf_min':
        return num(this.confMin);
      case 'conf_max':
        return num(this.confMax);
      case 'min_area':
        return num(this.minArea);
      case 'max_area':
        return num(this.maxArea);
      case 'max_rank':
        return num(this.maxRank);
      case 'origin':
        return this.origin;
      case 'embedding_state':
        return this.embeddingState;
      case 'review_status':
        return this.reviewStatus;
      case 'open_vocab_set':
        return this.openVocabSet ?? '';
      case 'source_prompt':
        return this.sourcePrompt ?? '';
      default:
        return undefined;
    }
  }

  /** Inverse of `valueOf`; an unparsable number or an enum value the
   *  contract does not list is dropped. */
  setValue(param: string, value: string | string[]): void {
    const list = Array.isArray(value) ? value : value === '' ? [] : [value];
    const str = Array.isArray(value) ? (value[0] ?? '') : value;
    const num = (): number | null => {
      if (str === '') return null;
      const n = Number(str);
      return Number.isFinite(n) ? n : null;
    };
    switch (param) {
      case 'class_name':
        this.classNames = list;
        break;
      case 'exclude_class_name':
        this.excludeClassNames = list;
        break;
      case 'conf_min':
        this.confMin = num();
        break;
      case 'conf_max':
        this.confMax = num();
        break;
      case 'min_area':
        this.minArea = num();
        break;
      case 'max_area':
        this.maxArea = num();
        break;
      case 'max_rank':
        this.maxRank = num();
        break;
      case 'origin':
        this.origin = pick(list, ITEM_ORIGINS);
        break;
      case 'embedding_state':
        this.embeddingState = pick(list, EMBEDDING_STATES);
        break;
      case 'review_status':
        this.reviewStatus = pick(list, REVIEW_STATUSES);
        break;
      case 'open_vocab_set':
        this.openVocabSet = str || null;
        break;
      case 'source_prompt':
        this.sourcePrompt = str || null;
        break;
    }
  }

  clear(): void {
    this.classNames = [];
    this.excludeClassNames = [];
    this.confMin = null;
    this.confMax = null;
    this.minArea = null;
    this.maxArea = null;
    this.maxRank = null;
    this.origin = [];
    this.embeddingState = [];
    this.reviewStatus = [];
    this.openVocabSet = null;
    this.sourcePrompt = null;
  }

  /** One removable chip per active value. */
  chips(): ItemFilterChip[] {
    const out: ItemFilterChip[] = [];
    for (const name of this.classNames) {
      out.push({
        param: 'class_name',
        label: name,
        clear: () => (this.classNames = this.classNames.filter((n) => n !== name)),
      });
    }
    for (const name of this.excludeClassNames) {
      out.push({
        param: 'exclude_class_name',
        label: `Not ${name}`,
        clear: () =>
          (this.excludeClassNames = this.excludeClassNames.filter((n) => n !== name)),
      });
    }
    const band = (
      param: string,
      label: string,
      lo: number | null,
      hi: number | null,
      clear: () => void,
    ): void => {
      if (lo == null && hi == null) return;
      out.push({ param, label: `${label}: ${lo ?? '…'} to ${hi ?? '…'}`, clear });
    };
    band('conf_min', 'Confidence', this.confMin, this.confMax, () => {
      this.confMin = null;
      this.confMax = null;
    });
    band('min_area', 'Area', this.minArea, this.maxArea, () => {
      this.minArea = null;
      this.maxArea = null;
    });
    if (this.maxRank != null) {
      out.push({
        param: 'max_rank',
        label: `Largest ${this.maxRank} per image`,
        clear: () => (this.maxRank = null),
      });
    }
    for (const v of this.origin) {
      out.push({
        param: 'origin',
        label: `Origin: ${humanizeId(v)}`,
        clear: () => (this.origin = this.origin.filter((x) => x !== v)),
      });
    }
    for (const v of this.embeddingState) {
      out.push({
        param: 'embedding_state',
        label: `Embedding: ${humanizeId(v)}`,
        clear: () => (this.embeddingState = this.embeddingState.filter((x) => x !== v)),
      });
    }
    for (const v of this.reviewStatus) {
      out.push({
        param: 'review_status',
        label: `Review: ${humanizeId(v)}`,
        clear: () => (this.reviewStatus = this.reviewStatus.filter((x) => x !== v)),
      });
    }
    if (this.openVocabSet) {
      out.push({
        param: 'open_vocab_set',
        label: `Open-vocabulary set: ${this.openVocabSet}`,
        clear: () => (this.openVocabSet = null),
      });
    }
    if (this.sourcePrompt) {
      out.push({
        param: 'source_prompt',
        label: `Prompt: ${this.sourcePrompt}`,
        clear: () => (this.sourcePrompt = null),
      });
    }
    return out;
  }
}

const URL_KEYS = [
  'class_name',
  'exclude_class_name',
  'conf_min',
  'conf_max',
  'min_area',
  'max_area',
  'max_rank',
  'origin',
  'embedding_state',
  'review_status',
  'open_vocab_set',
  'source_prompt',
] as const;
