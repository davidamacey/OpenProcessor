/**
 * The served JSON Schema of a preset body (pydantic `model_json_schema()`, draft 2020-12) read as a
 * list of fields for the help beside the editor: path, type, default, choices, limits and the
 * server's description. This only reads what the server wrote down; it enforces nothing (the server
 * judges the body on Save).
 */
import type { JsonSchemaNode, PresetSchemaWire } from '$lib/types_presets';

export interface SchemaField {
  /** `refresh.max_deferral_s`; `servers[].name` for a list of objects. */
  path: string;
  depth: number;
  type: string;
  required: boolean;
  /** JSON text of the default; null when the field has none. */
  default: string | null;
  choices: string[];
  /** "at least 0", "at most 1" ... */
  limits: string[];
  description: string;
}

type Defs = Record<string, JsonSchemaNode>;

function resolve(node: JsonSchemaNode, defs: Defs): JsonSchemaNode {
  const ref = node.$ref;
  if (!ref) return node;
  const target = defs[ref.slice('#/$defs/'.length)] ?? {};
  // Siblings of a $ref (the field's own description, default) win over the model's.
  const { $ref: _ref, ...own } = node;
  return { ...target, ...own };
}

/** The non-null alternatives of an `anyOf`, and whether null was one. */
function alternatives(node: JsonSchemaNode, defs: Defs): [JsonSchemaNode[], boolean] {
  if (!node.anyOf) return [[resolve(node, defs)], false];
  const live = node.anyOf.filter((n) => n.type !== 'null').map((n) => resolve(n, defs));
  return [live, live.length !== node.anyOf.length];
}

function typeLabel(node: JsonSchemaNode, defs: Defs): string {
  const [alts, nullable] = alternatives(node, defs);
  const parts = alts.map((a) => {
    if (a.enum) return 'choice';
    if (a.type === 'array') {
      const [inner] = alternatives(a.items ?? {}, defs);
      return `list of ${typeLabel(inner[0] ?? {}, defs)}`;
    }
    if (a.type === 'object' && !a.properties) {
      const ap = a.additionalProperties;
      const v = typeof ap === 'object' ? typeLabel(ap, defs) : 'value';
      return `map of ${v}`;
    }
    if (a.type === 'object') return 'object';
    return a.type ?? (a.const !== undefined ? 'constant' : 'value');
  });
  return [...new Set(parts)].join(' or ') + (nullable ? ' or null' : '');
}

function limitsOf(node: JsonSchemaNode): string[] {
  const out: string[] = [];
  if (node.minimum !== undefined) out.push(`at least ${node.minimum}`);
  if (node.exclusiveMinimum !== undefined) out.push(`above ${node.exclusiveMinimum}`);
  if (node.maximum !== undefined) out.push(`at most ${node.maximum}`);
  if (node.exclusiveMaximum !== undefined) out.push(`below ${node.exclusiveMaximum}`);
  if (node.minLength !== undefined) out.push(`length at least ${node.minLength}`);
  if (node.minItems !== undefined)
    out.push(`at least ${node.minItems} item${node.minItems === 1 ? '' : 's'}`);
  return out;
}

function choicesOf(node: JsonSchemaNode, defs: Defs): string[] {
  const [alts] = alternatives(node, defs);
  const values = alts.flatMap((a) => a.enum ?? (a.const !== undefined ? [a.const] : []));
  return values.map((v) => String(v));
}

function walk(
  node: JsonSchemaNode,
  defs: Defs,
  path: string,
  depth: number,
  out: SchemaField[],
): void {
  const here = resolve(node, defs);
  const required = new Set(here.required ?? []);
  for (const [name, raw] of Object.entries(here.properties ?? {})) {
    const prop = resolve(raw, defs);
    const [alts] = alternatives(raw, defs);
    const first = alts[0] ?? prop;
    const full = path ? `${path}.${name}` : name;
    out.push({
      path: full,
      depth,
      type: typeLabel(raw, defs),
      required: required.has(name),
      default: 'default' in prop ? JSON.stringify(prop.default) : null,
      choices: choicesOf(raw, defs),
      limits: limitsOf(first),
      description: prop.description ?? first.description ?? '',
    });
    if (first.properties) walk(first, defs, full, depth + 1, out);
    else if (first.type === 'array' && first.items) {
      const [inner] = alternatives(first.items, defs);
      if (inner[0]?.properties) walk(inner[0], defs, `${full}[]`, depth + 1, out);
    }
  }
}

/** One row per field at every depth, in the schema's own order. */
export function schemaFields(schema: PresetSchemaWire): SchemaField[] {
  const root = schema.json_schema;
  const out: SchemaField[] = [];
  walk(root, (root.$defs ?? {}) as Defs, '', 0, out);
  return out;
}
