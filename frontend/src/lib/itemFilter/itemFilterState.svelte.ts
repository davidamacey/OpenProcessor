/**
 * The operator's shared item-filter choices (OpenProcessor v0.4.0): one
 * state object read by the Workspace (every view, Review included) and
 * /export. It only holds what the operator picked and spells it in the two
 * wire forms (query parameters for list/stats routes, the `ItemFilter` body
 * for `selection` / `export/yolo item_filter`); the server decides what a
 * filter matches, how many items that is and which values are valid.
 */
import { humanizeId } from '$lib/humanizeId';
import type { GeoBox } from '$lib/types_map';
import {
  EMBEDDING_STATES,
  AUDIT_OUTCOMES,
  ITEM_ORIGINS,
  REVIEW_STATUSES,
  TEXT_MATCHES,
  type AuditOutcome,
  type EmbeddingState,
  type ItemFilter,
  type ItemFilterQuery,
  type ItemOrigin,
  type ReviewStatus,
  type TextMatch,
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

const formatBox = (b: GeoBox): string => `${b.west},${b.south},${b.east},${b.north}`;
const formatRing = (ring: [number, number][]): string =>
  ring.map(([lon, lat]) => `${lon},${lat}`).join(';');

/** Reads back what `formatBox` writes; anything else is dropped (the server refuses it anyway). */
function parseBox(raw: string | null): GeoBox | null {
  const parts = (raw ?? '').split(',').map(Number);
  if (parts.length !== 4 || parts.some((n) => !Number.isFinite(n))) return null;
  const [west, south, east, north] = parts;
  return { west, south, east, north };
}

/** A box on the embedding scatter, `x1,y1,x2,y2`; an inverted or unparsable one is dropped. */
export type VizBox = [number, number, number, number];

function parseVizBox(raw: string | null): VizBox | null {
  const parts = (raw ?? '').split(',').map(Number);
  if (parts.length !== 4 || parts.some((n) => !Number.isFinite(n))) return null;
  const [x1, y1, x2, y2] = parts;
  return x1 <= x2 && y1 <= y2 ? [x1, y1, x2, y2] : null;
}

function parseRing(raw: string | null): [number, number][] | null {
  if (!raw) return null;
  const ring = raw.split(';').map((v) => v.split(',').map(Number));
  if (ring.some((p) => p.length !== 2 || p.some((n) => !Number.isFinite(n)))) return null;
  return ring.map(([lon, lat]) => [lon, lat]);
}

function pick<T extends string>(values: string[], allowed: readonly T[]): T[] {
  return values.filter((v): v is T => (allowed as readonly string[]).includes(v));
}

/** Query params that carry one number, and the field each sets. */
const NUMBER_PARAMS: Record<
  string,
  'confMin' | 'confMax' | 'minArea' | 'maxArea' | 'maxRank'
> = {
  conf_min: 'confMin',
  conf_max: 'confMax',
  min_area: 'minArea',
  max_area: 'maxArea',
  max_rank: 'maxRank',
};

function parseFiniteOrNull(str: string): number | null {
  if (str === '') return null;
  const n = Number(str);
  return Number.isFinite(n) ? n : null;
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
  /** Audited crops by who was right; set by the link from an audit report, not by a bar control. */
  auditOutcomes = $state<AuditOutcome[]>([]);
  openVocabSet = $state<string | null>(null);
  sourcePrompt = $state<string | null>(null);
  /** One dataset import: the items it labeled or proposed. */
  importId = $state<string | null>(null);
  /** Capture days, `YYYY-MM-DD`, inclusive; `''` = unset. */
  takenFrom = $state('');
  takenTo = $state('');
  /** Clusters of the served partition (OR). */
  clusterUids = $state<string[]>([]);
  /** An area on the map: a box or a ring, as drawn; the server snaps it to the location policy. */
  geoBbox = $state<GeoBox | null>(null);
  geoPolygon = $state<[number, number][] | null>(null);
  hasLocation = $state<boolean | null>(null);
  /** A box drawn on the Embedding view, in scatter coordinates. */
  vizBox = $state<VizBox | null>(null);
  /** Set by a scope (a near-duplicate group, one image), not by a bar control. */
  nearDupGroup = $state<string | null>(null);
  imageIds = $state<string[]>([]);
  /** How the box text search matches; null = the server default. */
  regionTextMatch = $state<TextMatch | null>(null);

  get isEmpty(): boolean {
    return Object.keys(this.toBody()).length === 0;
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
    put('audit_outcome', [...this.auditOutcomes], this.auditOutcomes.length > 0);
    put('open_vocab_set', this.openVocabSet, !!this.openVocabSet);
    put('source_prompt', this.sourcePrompt, !!this.sourcePrompt);
    put('import_id', this.importId, !!this.importId);
    put('taken_from', this.takenFrom, !!this.takenFrom);
    put('taken_to', this.takenTo, !!this.takenTo);
    put('cluster_uid', [...this.clusterUids], this.clusterUids.length > 0);
    put('geo_bbox', this.geoBbox && formatBox(this.geoBbox), this.geoBbox != null);
    put(
      'geo_polygon',
      this.geoPolygon && formatRing(this.geoPolygon),
      this.geoPolygon != null,
    );
    put('has_location', this.hasLocation, this.hasLocation != null);
    put('viz_box', this.vizBox?.join(','), this.vizBox != null);
    put('near_dup_group', this.nearDupGroup, this.nearDupGroup != null);
    put('image_id', [...this.imageIds], this.imageIds.length > 0);
    put('region_text_match', this.regionTextMatch, this.regionTextMatch != null);
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
    if (this.auditOutcomes.length > 0) b.audit_outcome = [...this.auditOutcomes];
    if (this.takenFrom) b.taken_from = this.takenFrom;
    if (this.takenTo) b.taken_to = this.takenTo;
    return { ...b, ...this.textBody(), ...this.areaBody() };
  }

  /** The open-vocabulary, prompt and import keys of the body. */
  private textBody(): ItemFilter {
    const b: ItemFilter = {};
    if (this.openVocabSet) b.open_vocab_set = this.openVocabSet;
    if (this.sourcePrompt) b.source_prompt = this.sourcePrompt;
    if (this.importId) b.import_id = this.importId;
    return b;
  }

  /** The cluster, area, location, group and image keys of the body. */
  private areaBody(): ItemFilter {
    const b: ItemFilter = {};
    if (this.clusterUids.length > 0) b.cluster_uids = [...this.clusterUids];
    if (this.geoBbox) b.geo_bbox = { ...this.geoBbox };
    if (this.geoPolygon) b.geo_polygon = this.geoPolygon.map(([lon, lat]) => [lon, lat]);
    if (this.hasLocation != null) b.has_location = this.hasLocation;
    if (this.vizBox) b.viz_box = [...this.vizBox];
    if (this.nearDupGroup) b.near_dup_group = this.nearDupGroup;
    if (this.imageIds.length > 0) b.image_ids = [...this.imageIds];
    if (this.regionTextMatch) b.region_text_match = this.regionTextMatch;
    return b;
  }

  /** Reads the same names `toUrl` writes (repeatable keys). */
  fromUrl(params: Pick<URLSearchParams, 'get' | 'getAll'>): void {
    const num = (k: string): number | null => {
      const v = params.get(k);
      if (v == null || v === '') return null;
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    };
    this.classNames = params
      .getAll('class_name')
      .filter((v, i, all) => all.indexOf(v) === i);
    this.excludeClassNames = params
      .getAll('exclude_class_name')
      .filter((v, i, all) => all.indexOf(v) === i);
    this.confMin = num('conf_min');
    this.confMax = num('conf_max');
    this.minArea = num('min_area');
    this.maxArea = num('max_area');
    this.maxRank = num('max_rank');
    this.origin = pick(params.getAll('origin'), ITEM_ORIGINS);
    this.embeddingState = pick(params.getAll('embedding_state'), EMBEDDING_STATES);
    this.reviewStatus = pick(params.getAll('review_status'), REVIEW_STATUSES);
    this.auditOutcomes = pick(params.getAll('audit_outcome'), AUDIT_OUTCOMES);
    this.openVocabSet = params.get('open_vocab_set') || null;
    this.sourcePrompt = params.get('source_prompt') || null;
    this.importId = params.get('import_id') || null;
    this.takenFrom = params.get('taken_from') ?? '';
    this.takenTo = params.get('taken_to') ?? '';
    this.clusterUids = params
      .getAll('cluster_uid')
      .filter((v, i, all) => all.indexOf(v) === i);
    this.geoBbox = parseBox(params.get('geo_bbox'));
    this.geoPolygon = parseRing(params.get('geo_polygon'));
    const located = params.get('has_location');
    this.hasLocation = located === 'true' ? true : located === 'false' ? false : null;
    this.vizBox = parseVizBox(params.get('viz_box'));
    this.regionTextMatch =
      pick(params.getAll('region_text_match'), TEXT_MATCHES)[0] ?? null;
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
      case 'taken_from':
        return this.takenFrom;
      case 'taken_to':
        return this.takenTo;
      default:
        return this.textValue(param) ?? this.textMatchValue(param);
    }
  }

  private textValue(param: string): string | undefined {
    const field = TEXT_FIELDS[param];
    return field ? (this[field] ?? '') : undefined;
  }

  private textMatchValue(param: string): string | undefined {
    return param === 'region_text_match' ? (this.regionTextMatch ?? '') : undefined;
  }

  private setTextMatch(param: string, list: string[]): void {
    if (param === 'region_text_match') {
      this.regionTextMatch = pick(list, TEXT_MATCHES)[0] ?? null;
    }
  }

  /** Inverse of `valueOf`; an unparsable number or an enum value the
   *  contract does not list is dropped. */
  setValue(param: string, value: string | string[]): void {
    const list = Array.isArray(value) ? value : value === '' ? [] : [value];
    const str = Array.isArray(value) ? (value[0] ?? '') : value;
    const numberField = Object.hasOwn(NUMBER_PARAMS, param) ? NUMBER_PARAMS[param] : null;
    if (numberField) {
      this[numberField] = parseFiniteOrNull(str);
      return;
    }
    switch (param) {
      case 'class_name':
        this.classNames = list;
        break;
      case 'exclude_class_name':
        this.excludeClassNames = list;
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
      case 'taken_from':
        this.takenFrom = str;
        break;
      case 'taken_to':
        this.takenTo = str;
        break;
      default:
        this.setText(param, str);
        this.setTextMatch(param, list);
    }
  }

  private setText(param: string, str: string): void {
    const field = TEXT_FIELDS[param];
    if (field) this[field] = str || null;
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
    this.auditOutcomes = [];
    this.openVocabSet = null;
    this.sourcePrompt = null;
    this.importId = null;
    this.takenFrom = '';
    this.takenTo = '';
    this.clusterUids = [];
    this.geoBbox = null;
    this.geoPolygon = null;
    this.hasLocation = null;
    this.vizBox = null;
    this.nearDupGroup = null;
    this.imageIds = [];
    this.regionTextMatch = null;
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
    for (const v of this.auditOutcomes) {
      out.push({
        param: 'audit_outcome',
        label: `Audit: ${humanizeId(v)}`,
        clear: () => (this.auditOutcomes = this.auditOutcomes.filter((x) => x !== v)),
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
    if (this.importId) {
      out.push({
        param: 'import_id',
        label: `Import: ${this.importId}`,
        clear: () => (this.importId = null),
      });
    }
    if (this.takenFrom || this.takenTo) {
      out.push({
        param: 'taken_from',
        label: `Taken: ${this.takenFrom || '…'} to ${this.takenTo || '…'}`,
        clear: () => {
          this.takenFrom = '';
          this.takenTo = '';
        },
      });
    }
    return [...out, ...this.scopeChips()];
  }

  /** Chips for the cluster, area, location and scope values. */
  private scopeChips(): ItemFilterChip[] {
    const out: ItemFilterChip[] = [];
    for (const uid of this.clusterUids) {
      out.push({
        param: 'cluster_uid',
        label: `Cluster ${uid.slice(0, 8)}`,
        clear: () => (this.clusterUids = this.clusterUids.filter((u) => u !== uid)),
      });
    }
    if (this.geoBbox) {
      out.push({
        param: 'geo_bbox',
        label: 'Area: box',
        clear: () => (this.geoBbox = null),
      });
    }
    if (this.geoPolygon) {
      out.push({
        param: 'geo_polygon',
        label: 'Area: drawn shape',
        clear: () => (this.geoPolygon = null),
      });
    }
    if (this.hasLocation != null) {
      out.push({
        param: 'has_location',
        label: this.hasLocation ? 'With a location' : 'Without a location',
        clear: () => (this.hasLocation = null),
      });
    }
    if (this.vizBox) {
      out.push({
        param: 'viz_box',
        label: 'Area: embedding box',
        clear: () => (this.vizBox = null),
      });
    }
    if (this.nearDupGroup) {
      out.push({
        param: 'near_dup_group',
        label: 'Near-duplicate group',
        clear: () => (this.nearDupGroup = null),
      });
    }
    if (this.imageIds.length > 0) {
      out.push({
        param: 'image_id',
        label: `${this.imageIds.length} image${this.imageIds.length === 1 ? '' : 's'}`,
        clear: () => (this.imageIds = []),
      });
    }
    return out;
  }
}

/** The free-text filters: the query parameter and the field that holds it. */
const TEXT_FIELDS: Record<string, 'openVocabSet' | 'sourcePrompt' | 'importId'> = {
  open_vocab_set: 'openVocabSet',
  source_prompt: 'sourcePrompt',
  import_id: 'importId',
};

export const URL_KEYS = [
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
  'audit_outcome',
  'open_vocab_set',
  'source_prompt',
  'import_id',
  'taken_from',
  'taken_to',
  'cluster_uid',
  'geo_bbox',
  'geo_polygon',
  'has_location',
  'viz_box',
  'region_text_match',
] as const;
