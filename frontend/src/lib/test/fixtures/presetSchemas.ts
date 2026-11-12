/**
 * The schemas the server serves at `GET .../<kind>/schema`, as `model_json_schema()` and `kind_axes()`
 * wrote them (copied from the models; the contract test pins the route, the help reads these).
 */
import type { PresetKindId, PresetSchemaWire } from '$lib/types_presets';

export const PRESET_SCHEMAS = {
  serving_profile: {
    kind: 'serving_profile',
    axes: [],
    json_schema: {
      $defs: {
        ServingImagePrep: {
          additionalProperties: false,
          description: 'How images are decoded and resized before inference.',
          properties: {
            instances: {
              description: 'Image-prep model instances.',
              minimum: 1,
              title: 'Instances',
              type: 'integer',
            },
            workers: {
              description: 'Worker threads per instance.',
              minimum: 1,
              title: 'Workers',
              type: 'integer',
            },
            max_in_flight: {
              description: 'Images prepared at once before callers wait.',
              minimum: 1,
              title: 'Max In Flight',
              type: 'integer',
            },
            device_budget_mib: {
              description: 'Device memory the image prep may use, in MiB.',
              minimum: 0,
              title: 'Device Budget Mib',
              type: 'integer',
            },
            min_megapixels: {
              default: 0,
              description: 'Images smaller than this are prepared on the CPU.',
              minimum: 0,
              title: 'Min Megapixels',
              type: 'number',
            },
          },
          required: ['instances', 'workers', 'max_in_flight', 'device_budget_mib'],
          title: 'ServingImagePrep',
          type: 'object',
        },
        ServingPriorities: {
          additionalProperties: false,
          description:
            'Triton request priority per traffic class (a lower number is served first).',
          properties: {
            interactive: {
              description: 'Priority of requests a person is waiting on.',
              minimum: 0,
              title: 'Interactive',
              type: 'integer',
            },
            bulk: {
              description: 'Priority of ingest and batch work.',
              minimum: 0,
              title: 'Bulk',
              type: 'integer',
            },
          },
          required: ['interactive', 'bulk'],
          title: 'ServingPriorities',
          type: 'object',
        },
        ServingServer: {
          additionalProperties: false,
          description: 'One Triton server, one per card.',
          properties: {
            name: {
              description: 'The name `load_list` refers to; unique.',
              minLength: 1,
              title: 'Name',
              type: 'string',
            },
            grpc_url: {
              description: 'host:port clients send inference to.',
              minLength: 1,
              title: 'Grpc Url',
              type: 'string',
            },
            metrics_url: {
              description: 'host:port of its metrics endpoint.',
              minLength: 1,
              title: 'Metrics Url',
              type: 'string',
            },
            gpu_uuids: {
              description: 'NVML UUIDs of the cards it serves.',
              items: {
                type: 'string',
              },
              minItems: 1,
              title: 'Gpu Uuids',
              type: 'array',
            },
            numa_node: {
              anyOf: [
                {
                  minimum: 0,
                  type: 'integer',
                },
                {
                  type: 'null',
                },
              ],
              default: null,
              description: 'The NUMA node it is pinned to; null = not pinned.',
              title: 'Numa Node',
            },
          },
          required: ['name', 'grpc_url', 'metrics_url', 'gpu_uuids'],
          title: 'ServingServer',
          type: 'object',
        },
      },
      additionalProperties: false,
      properties: {
        servers: {
          description: 'The Triton servers, one per card.',
          items: {
            $ref: '#/$defs/ServingServer',
          },
          minItems: 1,
          title: 'Servers',
          type: 'array',
        },
        load_list: {
          additionalProperties: {
            items: {
              type: 'string',
            },
            type: 'array',
          },
          description: 'Per server name, the models it keeps resident.',
          title: 'Load List',
          type: 'object',
        },
        instances: {
          additionalProperties: {
            type: 'integer',
          },
          description: 'Instances per model.',
          title: 'Instances',
          type: 'object',
        },
        image_prep: {
          $ref: '#/$defs/ServingImagePrep',
          description: 'Image decode and resize settings.',
        },
        priorities: {
          $ref: '#/$defs/ServingPriorities',
          description: 'Request priorities per traffic class.',
        },
      },
      required: ['servers', 'load_list', 'instances', 'image_prep', 'priorities'],
      title: 'ServingProfileBody',
      type: 'object',
    },
  },
  cluster_profile: {
    kind: 'cluster_profile',
    axes: [
      {
        axis: 'clustering_item',
        label: 'Items',
        target: 'item',
      },
      {
        axis: 'clustering_region_box',
        label: 'Region boxes',
        target: 'region_box',
      },
    ],
    json_schema: {
      $defs: {
        ClusterRefreshRule: {
          additionalProperties: false,
          properties: {
            growth_threshold: {
              description: 'Growth since the last run that starts one.',
              exclusiveMinimum: 0,
              title: 'Growth Threshold',
              type: 'number',
            },
            max_deferral_s: {
              description: 'Longest a run waits for ingest to go quiet, in seconds.',
              minimum: 0,
              title: 'Max Deferral S',
              type: 'integer',
            },
            retry_backoff_s: {
              description:
                'Wait before retrying a run that failed or was cancelled, in seconds.',
              minimum: 0,
              title: 'Retry Backoff S',
              type: 'integer',
            },
          },
          required: ['growth_threshold', 'max_deferral_s', 'retry_backoff_s'],
          title: 'ClusterRefreshRule',
          type: 'object',
        },
      },
      additionalProperties: false,
      properties: {
        target: {
          description:
            'What is clustered; it must match the axis the profile is activated on.',
          enum: ['item', 'region_box'],
          title: 'Target',
          type: 'string',
        },
        level: {
          description: "The partition level, 'class' by default.",
          minLength: 1,
          title: 'Level',
          type: 'string',
        },
        embedding_field: {
          description: 'The vector field (the primary space of the target) to cluster.',
          minLength: 1,
          title: 'Embedding Field',
          type: 'string',
        },
        method: {
          description: 'The clustering method.',
          enum: ['ivf', 'minibatch_kmeans'],
          title: 'Method',
          type: 'string',
        },
        params: {
          additionalProperties: {
            anyOf: [
              {
                type: 'number',
              },
              {
                type: 'integer',
              },
              {
                type: 'string',
              },
              {
                type: 'boolean',
              },
              {
                type: 'null',
              },
            ],
          },
          description: 'Options of the method, such as n_clusters.',
          title: 'Params',
          type: 'object',
        },
        pool: {
          additionalProperties: {
            anyOf: [
              {
                type: 'number',
              },
              {
                type: 'integer',
              },
              {
                type: 'string',
              },
              {
                type: 'boolean',
              },
              {
                type: 'null',
              },
            ],
          },
          description: 'Stored with the profile; the fit does not read it yet.',
          title: 'Pool',
          type: 'object',
        },
        assign: {
          additionalProperties: {
            anyOf: [
              {
                type: 'number',
              },
              {
                type: 'integer',
              },
              {
                type: 'string',
              },
              {
                type: 'boolean',
              },
              {
                type: 'null',
              },
            ],
          },
          description: 'Stored with the profile; the fit does not read it yet.',
          title: 'Assign',
          type: 'object',
        },
        refresh: {
          $ref: '#/$defs/ClusterRefreshRule',
          description: 'When a new run starts.',
        },
        activation: {
          default: 'human',
          description:
            'auto_when_safe: a new partition becomes serving when its diff is safe; human: a person activates it.',
          enum: ['auto_when_safe', 'human'],
          title: 'Activation',
          type: 'string',
        },
        backend: {
          default: 'cpu',
          description: 'gpu_if_validated uses the GPU only where it was validated.',
          enum: ['cpu', 'gpu_if_validated'],
          title: 'Backend',
          type: 'string',
        },
      },
      required: ['target', 'level', 'embedding_field', 'method', 'refresh'],
      title: 'ClusterProfileBody',
      type: 'object',
    },
  },
  assist_loop_preset: {
    kind: 'assist_loop_preset',
    axes: [
      {
        axis: 'assist_loop_preset',
        label: 'Assist loop',
        target: null,
      },
    ],
    json_schema: {
      $defs: {
        LoopAudit: {
          additionalProperties: false,
          description:
            'A second model re-checks a sample of the labels the loop propagated.',
          properties: {
            vlm: {
              anyOf: [
                {
                  type: 'string',
                },
                {
                  type: 'null',
                },
              ],
              default: null,
              description: "The audit model's endpoint name; null = the audit is off.",
              title: 'Vlm',
            },
            acknowledge_external: {
              default: false,
              description: 'Confirms crops may go to an external audit endpoint.',
              title: 'Acknowledge External',
              type: 'boolean',
            },
            rate: {
              default: 0.01,
              description: 'Share of propagated labels re-checked.',
              exclusiveMinimum: 0,
              maximum: 1,
              title: 'Rate',
              type: 'number',
            },
            min_per_batch: {
              default: 30,
              description: 'Fewest labels re-checked per batch.',
              minimum: 1,
              title: 'Min Per Batch',
              type: 'integer',
            },
            max_crops: {
              default: 500,
              description: "The audit's own crop budget.",
              minimum: 1,
              title: 'Max Crops',
              type: 'integer',
            },
            floor: {
              default: 0.85,
              description: 'Agreement below which the audit fails.',
              exclusiveMaximum: 1,
              exclusiveMinimum: 0,
              title: 'Floor',
              type: 'number',
            },
            on_fail: {
              default: 'pause',
              description: 'What a failed audit does: pause the loop or roll back.',
              enum: ['pause', 'rollback'],
              title: 'On Fail',
              type: 'string',
            },
          },
          title: 'LoopAudit',
          type: 'object',
        },
        LoopBudget: {
          additionalProperties: false,
          description:
            'Hard caps; the loop never goes past one. ``max_vlm_crops`` has no default.',
          properties: {
            max_vlm_crops: {
              description: 'Crops the VLM may label in one run.',
              minimum: 1,
              title: 'Max Vlm Crops',
              type: 'integer',
            },
            max_vlm_requests: {
              anyOf: [
                {
                  minimum: 1,
                  type: 'integer',
                },
                {
                  type: 'null',
                },
              ],
              default: null,
              description: 'null = crops / images per call, plus the concurrency.',
              title: 'Max Vlm Requests',
            },
            max_wall_s: {
              anyOf: [
                {
                  exclusiveMinimum: 0,
                  type: 'number',
                },
                {
                  type: 'null',
                },
              ],
              default: null,
              description: 'null = 6 hours.',
              title: 'Max Wall S',
            },
          },
          required: ['max_vlm_crops'],
          title: 'LoopBudget',
          type: 'object',
        },
        LoopGate: {
          additionalProperties: false,
          description:
            'When a sampled cluster may pass its class to all its other members.',
          properties: {
            theta: {
              default: 0.9,
              description: 'Homogeneity the cluster must be shown to have.',
              maximum: 0.99,
              minimum: 0.7,
              title: 'Theta',
              type: 'number',
            },
            alpha: {
              default: 0.05,
              description: 'Chance allowed of passing a cluster below theta.',
              maximum: 0.1,
              minimum: 0.001,
              title: 'Alpha',
              type: 'number',
            },
            max_dissent: {
              default: 1,
              description: 'Sampled answers allowed to disagree with the top class.',
              maximum: 3,
              minimum: 0,
              title: 'Max Dissent',
              type: 'integer',
            },
          },
          title: 'LoopGate',
          type: 'object',
        },
        LoopRecursion: {
          additionalProperties: false,
          description: 'How a cluster that fails the gate is split and tried again.',
          properties: {
            max_depth: {
              default: 3,
              description: 'Levels a failing node is split.',
              maximum: 6,
              minimum: 0,
              title: 'Max Depth',
              type: 'integer',
            },
            branching: {
              default: 4,
              description: 'Parts a failing node is split into.',
              maximum: 8,
              minimum: 2,
              title: 'Branching',
              type: 'integer',
            },
            min_node_size: {
              default: 40,
              description: 'Smaller failing nodes are left for review.',
              minimum: 2,
              title: 'Min Node Size',
              type: 'integer',
            },
          },
          title: 'LoopRecursion',
          type: 'object',
        },
      },
      additionalProperties: false,
      properties: {
        strategy: {
          const: 'cluster_sampled',
          default: 'cluster_sampled',
          description: 'How crops are picked: one sample per cluster.',
          title: 'Strategy',
          type: 'string',
        },
        gate: {
          $ref: '#/$defs/LoopGate',
        },
        recursion: {
          $ref: '#/$defs/LoopRecursion',
        },
        budget: {
          $ref: '#/$defs/LoopBudget',
          description: 'Hard caps; the loop never goes past one.',
        },
        on_budget: {
          default: 'pause',
          description:
            "'pause': a reached limit pauses the loop; 'complete': the nodes the budget cannot cover are left unvisited.",
          enum: ['complete', 'pause'],
          title: 'On Budget',
          type: 'string',
        },
        audit: {
          $ref: '#/$defs/LoopAudit',
        },
        vlm_concurrency: {
          default: 16,
          description: 'VLM requests in flight at once.',
          maximum: 128,
          minimum: 1,
          title: 'Vlm Concurrency',
          type: 'integer',
        },
        node_concurrency: {
          default: 8,
          description: 'Cluster nodes worked on at once.',
          maximum: 32,
          minimum: 1,
          title: 'Node Concurrency',
          type: 'integer',
        },
      },
      required: ['budget'],
      title: 'AssistLoopPresetBody',
      type: 'object',
    },
  },
  near_dup_profile: {
    kind: 'near_dup_profile',
    axes: [
      {
        axis: 'near_dup',
        label: 'Near-duplicates',
        target: null,
      },
    ],
    json_schema: {
      $defs: {
        NearDupImageProfile: {
          additionalProperties: false,
          properties: {
            space_key: {
              anyOf: [
                {
                  type: 'string',
                },
                {
                  type: 'null',
                },
              ],
              default: null,
              description: 'A live space of the level; null: the primary space.',
              title: 'Space Key',
            },
            dedup_cosine: {
              default: 0.98,
              description: 'Images at or above it are duplicates.',
              maximum: 1.0,
              title: 'Dedup Cosine',
              type: 'number',
            },
            leak_cosine: {
              default: 0.98,
              description: 'Images at or above it must stay on one split side.',
              minimum: 0.5,
              title: 'Leak Cosine',
              type: 'number',
            },
            max_group_size: {
              default: 1000,
              description: 'Largest group kept whole; a bigger one is split.',
              minimum: 2,
              title: 'Max Group Size',
              type: 'integer',
            },
            k: {
              default: 64,
              description: 'Nearest neighbours searched per image.',
              maximum: 256,
              minimum: 8,
              title: 'K',
              type: 'integer',
            },
          },
          title: 'NearDupImageProfile',
          type: 'object',
        },
        NearDupItemProfile: {
          additionalProperties: false,
          properties: {
            space_key: {
              anyOf: [
                {
                  type: 'string',
                },
                {
                  type: 'null',
                },
              ],
              default: null,
              description: 'A live space of the level; null: the primary space.',
              title: 'Space Key',
            },
            dedup_cosine: {
              default: 0.98,
              description: 'Items at or above it are duplicates.',
              maximum: 1.0,
              title: 'Dedup Cosine',
              type: 'number',
            },
            leak_cosine: {
              default: 0.98,
              description: 'Items at or above it must stay on one split side.',
              minimum: 0.5,
              title: 'Leak Cosine',
              type: 'number',
            },
            max_group_size: {
              default: 1000,
              description: 'Largest group kept whole; a bigger one is split.',
              minimum: 2,
              title: 'Max Group Size',
              type: 'integer',
            },
            scope: {
              const: 'image_leak_groups',
              default: 'image_leak_groups',
              description:
                'Items are compared only inside the leak groups of their images.',
              title: 'Scope',
              type: 'string',
            },
            same_detector_class: {
              default: true,
              description: 'Compare only items of one detector class.',
              title: 'Same Detector Class',
              type: 'boolean',
            },
          },
          title: 'NearDupItemProfile',
          type: 'object',
        },
      },
      additionalProperties: false,
      properties: {
        version: {
          const: 1,
          default: 1,
          description: 'The body format version.',
          title: 'Version',
          type: 'integer',
        },
        image: {
          $ref: '#/$defs/NearDupImageProfile',
          description: 'Cuts for whole images.',
        },
        item: {
          $ref: '#/$defs/NearDupItemProfile',
          description: 'Cuts for items (region boxes).',
        },
      },
      title: 'NearDupProfileBody',
      type: 'object',
    },
  },
} as unknown as Record<PresetKindId, PresetSchemaWire>;
