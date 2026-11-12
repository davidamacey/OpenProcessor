/**
 * The icon set: Lucide (`@lucide/svelte`, ISC; ATTRIBUTION.md), one deep import per icon so
 * the bundle holds only these (the package's `icons/*` export). `ui/Icon.svelte` renders by
 * key. To add an icon, add its import and key here; keys are Lucide's own kebab-case names
 * (https://lucide.dev/icons).
 */
import type { LucideIcon } from '@lucide/svelte';
import ArrowDown from '@lucide/svelte/icons/arrow-down';
import ArrowLeft from '@lucide/svelte/icons/arrow-left';
import ArrowRight from '@lucide/svelte/icons/arrow-right';
import ArrowUp from '@lucide/svelte/icons/arrow-up';
import Ban from '@lucide/svelte/icons/ban';
import CaseSensitive from '@lucide/svelte/icons/case-sensitive';
import Check from '@lucide/svelte/icons/check';
import ChevronDown from '@lucide/svelte/icons/chevron-down';
import ChevronLeft from '@lucide/svelte/icons/chevron-left';
import ChevronRight from '@lucide/svelte/icons/chevron-right';
import ChevronUp from '@lucide/svelte/icons/chevron-up';
import ChevronsUpDown from '@lucide/svelte/icons/chevrons-up-down';
import Circle from '@lucide/svelte/icons/circle';
import CircleAlert from '@lucide/svelte/icons/circle-exclamation-point';
import CircleCheck from '@lucide/svelte/icons/circle-check';
import CircleDot from '@lucide/svelte/icons/circle-dot';
import CircleX from '@lucide/svelte/icons/circle-x';
import Copy from '@lucide/svelte/icons/copy';
import CornerDownRight from '@lucide/svelte/icons/corner-down-right';
import Download from '@lucide/svelte/icons/download';
import Ellipsis from '@lucide/svelte/icons/ellipsis';
import ExternalLink from '@lucide/svelte/icons/external-link';
import Eye from '@lucide/svelte/icons/eye';
import EyeOff from '@lucide/svelte/icons/eye-off';
import FolderOpen from '@lucide/svelte/icons/folder-open';
import Funnel from '@lucide/svelte/icons/funnel';
import Group from '@lucide/svelte/icons/group';
import Hand from '@lucide/svelte/icons/hand';
import Info from '@lucide/svelte/icons/info';
import Layers2 from '@lucide/svelte/icons/layers-2';
import LoaderCircle from '@lucide/svelte/icons/loader-circle';
import Lock from '@lucide/svelte/icons/lock';
import Maximize from '@lucide/svelte/icons/maximize';
import Menu from '@lucide/svelte/icons/menu';
import Minus from '@lucide/svelte/icons/minus';
import MousePointer2 from '@lucide/svelte/icons/mouse-pointer-2';
import Pencil from '@lucide/svelte/icons/pencil';
import Plus from '@lucide/svelte/icons/plus';
import Redo2 from '@lucide/svelte/icons/redo-2';
import RefreshCw from '@lucide/svelte/icons/refresh-cw';
import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
import Scan from '@lucide/svelte/icons/scan';
import Search from '@lucide/svelte/icons/search';
import Settings from '@lucide/svelte/icons/settings';
import Square from '@lucide/svelte/icons/square';
import SquareDashed from '@lucide/svelte/icons/square-dashed';
import Trash from '@lucide/svelte/icons/trash';
import TriangleAlert from '@lucide/svelte/icons/triangle-exclamation-point';
import Type from '@lucide/svelte/icons/type';
import Undo2 from '@lucide/svelte/icons/undo-2';
import Upload from '@lucide/svelte/icons/upload';
import X from '@lucide/svelte/icons/x';
import ZoomIn from '@lucide/svelte/icons/zoom-in';
import ZoomOut from '@lucide/svelte/icons/zoom-out';

export const ICONS = {
  'arrow-down': ArrowDown,
  'arrow-left': ArrowLeft,
  'arrow-right': ArrowRight,
  'arrow-up': ArrowUp,
  ban: Ban,
  'case-sensitive': CaseSensitive,
  check: Check,
  'chevron-down': ChevronDown,
  'chevron-left': ChevronLeft,
  'chevron-right': ChevronRight,
  'chevron-up': ChevronUp,
  'chevrons-up-down': ChevronsUpDown,
  circle: Circle,
  'circle-alert': CircleAlert,
  'circle-check': CircleCheck,
  'circle-dot': CircleDot,
  'circle-x': CircleX,
  copy: Copy,
  'corner-down-right': CornerDownRight,
  download: Download,
  ellipsis: Ellipsis,
  'external-link': ExternalLink,
  eye: Eye,
  'eye-off': EyeOff,
  'folder-open': FolderOpen,
  funnel: Funnel,
  group: Group,
  hand: Hand,
  info: Info,
  'layers-2': Layers2,
  'loader-circle': LoaderCircle,
  lock: Lock,
  maximize: Maximize,
  menu: Menu,
  minus: Minus,
  'mouse-pointer-2': MousePointer2,
  pencil: Pencil,
  plus: Plus,
  'redo-2': Redo2,
  'refresh-cw': RefreshCw,
  'rotate-ccw': RotateCcw,
  scan: Scan,
  search: Search,
  settings: Settings,
  square: Square,
  'square-dashed': SquareDashed,
  trash: Trash,
  'triangle-alert': TriangleAlert,
  type: Type,
  'undo-2': Undo2,
  upload: Upload,
  x: X,
  'zoom-in': ZoomIn,
  'zoom-out': ZoomOut,
} satisfies Record<string, LucideIcon>;

export type IconName = keyof typeof ICONS;
