import { describe, expect, it } from 'vitest';
import { mlflowBaseUrl } from './mlflowLink';

describe('mlflowBaseUrl (visual audit T1)', () => {
  it('uses the origin of the first served run URL, never a hardcoded port', () => {
    expect(
      mlflowBaseUrl([
        null,
        'http://localhost:4731/#/experiments/1/runs/66e3097b44c348ceaca8416c4eabc4d5',
      ]),
    ).toBe('http://localhost:4731');
  });

  it('an explicit PUBLIC_MLFLOW_URL wins', () => {
    expect(mlflowBaseUrl(['http://localhost:4731/x'], 'https://mlflow.example')).toBe(
      'https://mlflow.example',
    );
  });

  it('nothing served means no link', () => {
    expect(mlflowBaseUrl([null, undefined, 'not a url'])).toBeNull();
    expect(mlflowBaseUrl([])).toBeNull();
  });
});
