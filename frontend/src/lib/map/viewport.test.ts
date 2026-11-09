import { describe, expect, it } from 'vitest';
import { viewQuery } from './viewport';

describe('viewQuery', () => {
  it('sends the box and the whole zoom, capped at the served deepest zoom', () => {
    expect(viewQuery({ west: 7, south: 46, east: 9, north: 48 }, 6.9, 12)).toEqual({
      zoom: 6,
      bbox: '7,46,9,48',
    });
    expect(viewQuery({ west: 7, south: 46, east: 9, north: 48 }, 16.2, 12).zoom).toBe(12);
    expect(viewQuery({ west: 7, south: 46, east: 9, north: 48 }, -1, 12).zoom).toBe(0);
  });

  it('sends no box for a view that spans the world', () => {
    expect(viewQuery({ west: -200, south: -80, east: 200, north: 80 }, 1, 12)).toEqual({
      zoom: 1,
    });
  });

  it('wraps a longitude outside -180..180 so a box across the antimeridian is west > east', () => {
    expect(viewQuery({ west: 170, south: -10, east: 190, north: 10 }, 4, 12).bbox).toBe(
      '170,-10,-170,10',
    );
    expect(viewQuery({ west: -190, south: -10, east: -170, north: 10 }, 4, 12).bbox).toBe(
      '170,-10,-170,10',
    );
  });

  it('clamps latitudes the map can show past the poles', () => {
    expect(viewQuery({ west: 0, south: -95, east: 10, north: 95 }, 2, 12).bbox).toBe(
      '0,-90,10,90',
    );
  });
});
