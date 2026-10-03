import { describe, expect, it } from 'vitest';
import { externalHref } from './mlflowLink';

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
