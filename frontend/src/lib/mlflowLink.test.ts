import { describe, expect, it } from 'vitest';
import { externalHref, mlflowBaseUrl } from './mlflowLink';

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

describe('externalHref', () => {
  it('keeps an http(s) URL and drops every other scheme', () => {
    expect(externalHref('http://op-mlflow:5000/#/runs/1')).toBe(
      'http://op-mlflow:5000/#/runs/1',
    );
    expect(externalHref('https://mlflow.example/r')).toBe('https://mlflow.example/r');
    expect(externalHref('javascript:alert(1)')).toBeNull();
    expect(externalHref('data:text/html,x')).toBeNull();
    expect(externalHref('/relative')).toBeNull();
    expect(externalHref(null)).toBeNull();
    expect(externalHref('')).toBeNull();
  });
});
