import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Usage: set environment vars for API_BASE (default http://localhost:8081), ADMIN_USER, ADMIN_PASS
const API_BASE = __ENV.API_BASE || 'http://localhost:8081';
const ADMIN_USER = __ENV.ADMIN_USER || 'admin';
const ADMIN_PASS = __ENV.ADMIN_PASS || 'adminpass';
const MODEL_PATH = __ENV.MODEL_PATH || '/v1/models/multimodal_demo/versions/v1/predict';

export let errorRate = new Rate('errors');

export let options = {
  stages: [
    { duration: '1m', target: 10 },   // warmup
    { duration: '3m', target: 50 },   // ramp to medium load
    { duration: '5m', target: 100 },  // sustained load
    { duration: '2m', target: 0 },    // ramp down
  ],
  thresholds: {
    errors: ['rate<0.01'], // fail if >1% errors
    http_req_duration: ['p(95)<2000'], // p95 < 2000ms - tune for your SLO
  },
};

function get_token() {
  const url = `${API_BASE}/auth/token`;
  const payload = {
    username: ADMIN_USER,
    password: ADMIN_PASS,
  };
  // OAuth2 password grant uses form-urlencoded
  const headers = { 'Content-Type': 'application/x-www-form-urlencoded' };
  let res = http.post(url, `username=${encodeURIComponent(payload.username)}&password=${encodeURIComponent(payload.password)}&grant_type=password`, { headers: headers });
  if (check(res, { 'obtained token': (r) => r.status === 200 && r.json('access_token') })) {
    return res.json('access_token');
  }
  errorRate.add(1);
  return null;
}

export default function () {
  group('Auth -> Predict -> Jobs -> Billing', function () {
    // get token once per VU iteration (k6 will spawn many VUs)
    const token = get_token();
    if (!token) {
      sleep(1);
      return;
    }
    const headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };

    // 1) Predict (multimodal text payload)
    const predictPayload = JSON.stringify({ text: "Hello world. Please summarize and include safe content." });
    let r1 = http.post(`${API_BASE}${MODEL_PATH}`, predictPayload, { headers: headers, timeout: '120s' });
    check(r1, {
      'predict status 2xx or 404 (optional)': (r) => r.status === 200 || r.status === 404,
      'predict returned quickly': (r) => r.timings.duration < 5000
    }) || errorRate.add(1);

    // 2) Enqueue a job (to stress Celery / queue + billing)
    const jobPayload = JSON.stringify({ work_units: 1, parameters: { test: true } });
    let r2 = http.post(`${API_BASE}/v1/jobs`, jobPayload, { headers: headers });
    check(r2, { 'job enqueued or accepted': (r) => r.status === 202 || r.status === 201 }) || errorRate.add(1);

    // 3) Billing read (list invoices) - exercise tenant/billing throttle path
    let r3 = http.get(`${API_BASE}/v1/billing/invoices`, { headers: headers });
    check(r3, { 'billing list ok': (r) => r.status === 200 }) || errorRate.add(1);

    // small sleep to moderate pacing between actions
    sleep(Math.random() * 1.5);
  });
}
