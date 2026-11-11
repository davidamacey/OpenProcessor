import base from './eslint.config.js';

/** Used only by scripts/codegen/check_frontend_complexity.py; see baselines/frontend_complexity.txt. */
export default [...base, { rules: { complexity: ['error', { max: 15 }] } }];
