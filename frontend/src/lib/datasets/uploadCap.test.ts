import { afterEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_DATASET_UPLOAD_MAX_MB,
  MULTIPART_ALLOWANCE_BYTES,
  datasetUploadMaxBytes,
  proxyUploadMaxBytes,
} from './uploadCap';

const MB = 1024 * 1024;
const w = window as unknown as Record<string, unknown>;

afterEach(() => {
  delete w.__CROPWRIGHT_DATASET_UPLOAD_MAX_MB__;
});

describe('uploadCap', () => {
  it('leaves room for the multipart envelope under the nginx body cap', () => {
    // The request body is the archive plus its multipart framing; a file
    // exactly at the nginx limit would be refused by nginx's HTML 413.
    expect(proxyUploadMaxBytes()).toBe(
      DEFAULT_DATASET_UPLOAD_MAX_MB * MB - MULTIPART_ALLOWANCE_BYTES,
    );
    expect(MULTIPART_ALLOWANCE_BYTES).toBeGreaterThan(0);
  });

  it('reads the deployment cap from the runtime global', () => {
    w.__CROPWRIGHT_DATASET_UPLOAD_MAX_MB__ = '100';
    expect(proxyUploadMaxBytes()).toBe(100 * MB - MULTIPART_ALLOWANCE_BYTES);
  });

  it('control: the lower of the served cap and the proxy cap wins', () => {
    expect(datasetUploadMaxBytes(5 * MB)).toBe(5 * MB);
    expect(datasetUploadMaxBytes(Number.MAX_SAFE_INTEGER)).toBe(proxyUploadMaxBytes());
  });
});
