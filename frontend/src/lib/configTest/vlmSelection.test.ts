import { describe, expect, it } from 'vitest';
import { fromTestVlmSelection, toTestVlmSelection } from './vlmSelection';

describe('test VLM selection', () => {
  it('the default pick sends nothing', () => {
    expect(toTestVlmSelection({ vlm: null, acknowledgeExternal: false })).toBeNull();
    // An acknowledgement without an endpoint is meaningless: still nothing.
    expect(toTestVlmSelection({ vlm: null, acknowledgeExternal: true })).toBeNull();
  });

  it('names the endpoint with a null revision and no acknowledgement unless ticked', () => {
    expect(toTestVlmSelection({ vlm: 'local', acknowledgeExternal: false })).toEqual({
      vlm_name: 'local',
      vlm_revision: null,
    });
    expect(toTestVlmSelection({ vlm: 'cloud', acknowledgeExternal: true })).toEqual({
      vlm_name: 'cloud',
      vlm_revision: null,
      acknowledge_external: true,
    });
  });

  it('reads a selection back into the picker state', () => {
    expect(fromTestVlmSelection(null)).toEqual({ vlm: null, acknowledgeExternal: false });
    expect(
      fromTestVlmSelection({ vlm_name: 'cloud', acknowledge_external: true }),
    ).toEqual({ vlm: 'cloud', acknowledgeExternal: true });
  });
});
