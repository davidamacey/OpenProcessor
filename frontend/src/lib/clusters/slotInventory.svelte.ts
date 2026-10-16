/**
 * Synthetic slot inventory cards for the /clusters grid: one card per
 * registered slot with a browse endpoint, since a slot's regions are
 * sub-boxes on other items and never form cluster docs of their own.
 */
import { resolveApiUrl } from '$lib/api';
import { getRegions } from '$lib/api_regions';
import { registeredSlots } from '$lib/annotations/registeredSlots';
import { displayBoxOf } from '$lib/annotations/rowBox';
import type { SlotSpec } from '$lib/annotations/types';
import { findActiveClassByName } from '$lib/classNameKey';
import type { Cluster } from '$lib/types';
import { classesStore } from '$stores/classes.svelte';

export function createSlotInventory() {
  // Synthetic slot inventory cards. A slot's regions are sub-bboxes on
  // other items, not cluster docs, so the cluster grid never produces a
  // card for them. We surface one per slot from its browse endpoint so the
  // operator can click into the slot's inventory the same way they click
  // into any other class cluster. Empty until the first browse call
  // resolves; the cluster grid shows none during that window.
  let slotInventoryCards = $state<Cluster[]>([]);

  // Build one synthetic inventory card per registered slot that has a
  // browse endpoint and whose bound class is in the registry. A slot's
  // regions live as sub-boxes on items of other classes (not cluster
  // docs), so the cluster grid never includes them. Each card is built
  // from its own slot's `queue.browsePath` (total + the first 4 boxed
  // items as thumbnails) and bound to its own class; a slot with no
  // browse endpoint gets no card. The grid renders the cards first on
  // the unfiltered view.
  async function loadSlotInventoryCards(): Promise<void> {
    const cards: Cluster[] = [];
    for (const slot of registeredSlots) {
      const card = await buildSlotInventoryCard(slot);
      if (card) cards.push(card);
    }
    if (cards.length > 0 || slotInventoryCards.length > 0) slotInventoryCards = cards;
  }

  async function buildSlotInventoryCard(slot: SlotSpec): Promise<Cluster | null> {
    const browsePath = slot.capabilities.queue?.browsePath;
    const className = slot.bind.className;
    if (!browsePath || !className) return null;
    const cls = findActiveClassByName(classesStore.classes, className);
    if (!cls) return null;
    try {
      // Pull a slightly larger window than 4 so we can drop items
      // missing a sub-box without falling below the tile count.
      const res = await getRegions(browsePath, {
        page: 1,
        page_size: 12,
      });
      // A tile is one box's close-up, so only rows with a served box
      // thumbnail qualify.
      const reps = res.items
        .map((p) => ({ p, box: displayBoxOf(p.slots?.[slot.key], p.region_box_id) }))
        .filter((r) => r.box?.thumbnailUrl != null)
        .slice(0, 4);
      return {
        id: cls.id,
        // Not a real cluster: cluster_kind/purity_tier/promotable/etc.
        // have no server-served value for a slot-inventory entry, so
        // they're left at honest defaults rather than invented. See
        // `isSlotCard` on the type and its use in purityBadge/borderColor
        // in ClusterGridCard, which skip the purity badge entirely for this card.
        cluster_kind: 'unassigned',
        validated_count: 0,
        size: res.total,
        purity: null,
        purity_tier: null,
        promotable: false,
        core_similarity_min: null,
        is_unlabeled: false,
        dominant_class_id: cls.id,
        dominant_class_name: cls.name,
        dominant_pct: null,
        n_subclusters: 0,
        subcluster_ids: [],
        has_subclusters: false,
        representative_crop_ids: reps.map((r) => r.p.crop_id),
        // Region close-ups, not parent-item thumbnails: the card is the
        // entry point to the slot's own inventory.
        representative_thumb_urls: reps.map((r) => resolveApiUrl(r.box!.thumbnailUrl!)),
        updated_at: null,
        isSlotCard: true,
        slotDisplayName: slot.capabilities.queue?.tabLabel ?? slot.label.plural,
      };
    } catch {
      return null;
    }
  }

  return {
    get slotInventoryCards() {
      return slotInventoryCards;
    },
    loadSlotInventoryCards,
  };
}

export type SlotInventory = ReturnType<typeof createSlotInventory>;
