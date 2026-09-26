import { test, expect, Page } from '@playwright/test';

async function app(page: Page, modelPath: string | null = null) {
  let effect: any = { mode: 'onnx_faceswap', strength: .82, smoothing: .5, scale: 1.22, y_offset: .5, mirror: true, debug: false,
    target_image_path: null, model_path: modelPath, provider: 'cpu', precision: 'fp32', edge_feather: .35, color_match: .5,
    sharpen: .2, temporal_smoothing: .4, background_enabled: false, background_path: null, background_strength: 1, background_threshold: 1, background_smoothing: 0 };
  let revision = 1;
  let status: any = { phase: 'idle', generation: 0, running: false, fps_actual: 0, frames_processed: 0, frame_age_ms: 0, dropped_frames: 0, effect_revision: revision };
  let voice: any = { phase: 'idle', running: false, synthesis_latency_ms: 0 };
  const requests: Array<{ path: string; body: any }> = [];
  await page.route('**/preview.*', route => route.abort());
  await page.route('**/api/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    let body: any = {};
    if (request.method() === 'POST') {
      body = request.headers()['content-type']?.includes('application/json') ? request.postDataJSON() : {};
      requests.push({ path, body });
    }
    let result: any = {};
    if (path === '/api/status') result = status;
    if (path === '/api/session/config') result = { effect, revision };
    if (path === '/api/session/effect') { effect = body.effect; revision++; status.effect_revision = revision; result = status; }
    if (path === '/api/devices') result = [{ index: 0, device_id: 'rgb-id', label: 'Standard webcam', kind: 'standard', is_default: true }, { index: 2, device_id: 'ir-id', label: 'Infrared camera', kind: 'infrared' }, { index: 4, device_id: 'obs-id', label: 'Virtual: OBS Virtual Camera', kind: 'virtual' }];
    if (path === '/api/capabilities') result = { platform: 'Linux', onnxruntime: false, insightface: false, default_model_present: false, models_dir: '/user/models', v4l2loopback_devices: [], target_presets: [], voice_modes: ['dsp'], swap_backends: [{ id: 'inswapper', label: 'InSwapper', default_model: 'inswapper_128.onnx' }, { id: 'example', label: 'Example engine', default_model: 'example.onnx' }] };
    if (path === '/api/setup') result = { terms_url: 'https://example.test/terms', terms_accepted: false, completed: true, download_bytes: 843125212, required_free_bytes: 1200000000, components: [], ready: true };
    if (path === '/api/voice/status') result = voice;
    if (path === '/api/voice/devices') result = [];
    if (path === '/api/voice/virtual-mic') result = { source_present: false };
    if (path === '/api/session/start') {
      status = { ...status, generation: status.generation + 1, running: true, phase: 'running', active_capture: body, source_index: body.source_index ?? 0, virtual_camera: body.virtual_camera };
      result = status;
    }
    if (path === '/api/session/stop') { status = { ...status, running: false, phase: 'idle' }; result = status; }
    if (path === '/api/session/stop-all') {
      status = { ...status, running: false, phase: 'idle' }; voice = { ...voice, running: false, phase: 'idle' };
      result = { video: { status, error: null }, voice: { status: voice, error: null }, tts: { status: { phase: 'idle' }, error: null } };
    }
    await route.fulfill({ json: result });
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Start camera', exact: true })).toBeEnabled();
  return requests;
}

test('backend selection uses catalog and clears the previous model path', async ({ page }) => {
  const requests = await app(page, '/models/custom-inswapper.onnx');
  await page.locator('summary').filter({ hasText: /^Performance$/ }).click();
  await expect(page.getByLabel('Face-swap engine')).toHaveValue('inswapper');
  await page.getByLabel('Face-swap engine').selectOption('example');
  await expect.poll(() => requests.filter(r => r.path === '/api/session/effect').at(-1)?.body.effect.swap_backend).toBe('example');
  expect(requests.filter(r => r.path === '/api/session/effect').at(-1)?.body.effect.model_path).toBeNull();
});

test('camera changes show restart requirement and stop all is explicit', async ({ page }) => {
  const requests = await app(page);
  await page.getByRole('button', { name: 'Start camera', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Stop camera', exact: true })).toBeVisible();
  await page.locator('summary').filter({ hasText: /^Camera$/ }).click();
  await page.locator('#camera').selectOption('2');
  await expect(page.getByText('Changes apply on restart', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Apply & restart camera' }).click();
  await expect(page.getByText('Changes apply on restart', { exact: true })).toHaveCount(0);
  expect(requests.filter(r => r.path === '/api/session/start').at(-1)?.body.source_id).toBe('ir-id');
  await page.getByRole('button', { name: 'Stop all', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Start camera', exact: true })).toBeVisible();
});

test('removed options use fixed neural defaults', async ({ page }) => {
  const requests = await app(page);
  await expect(page.getByText('Setup & help', { exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Camera effect')).toHaveCount(0);
  await expect(page.getByLabel('Virtual camera')).toHaveCount(0);
  await expect(page.getByLabel('Precision')).toHaveCount(0);
  await expect(page.getByText('Color match', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Start camera', exact: true }).click();
  const started = requests.filter(r => r.path === '/api/session/start').at(-1)?.body;
  expect(started.effect.mode).toBe('onnx_faceswap');
  expect(started.effect.precision).toBe('fp32');
  expect(started.effect.color_match).toBe(0.5);
});

test('selecting a virtual camera enables virtual output automatically', async ({ page }) => {
  const requests = await app(page);
  await page.locator('summary').filter({ hasText: /^Camera$/ }).click();
  await page.locator('#camera').selectOption('4');
  await page.getByRole('button', { name: 'Start camera', exact: true }).click();
  const started = requests.filter(r => r.path === '/api/session/start').at(-1)?.body;
  expect(started.source_id).toBe('obs-id');
  expect(started.virtual_camera).toBe(true);
});

test('opening background picker makes no settings request', async ({ page }) => {
  const requests = await app(page);
  await page.locator('summary').filter({ hasText: /^Background$/ }).click();
  const chooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Choose image' }).click();
  await chooser;
  expect(requests.filter(r => r.path === '/api/session/effect')).toHaveLength(0);
});

for (const size of [{ width: 1440, height: 900 }, { width: 1024, height: 768 }, { width: 390, height: 844 }, { width: 320, height: 568 }]) {
  test(`expanded controls fit ${size.width}`, async ({ page }) => {
    await page.setViewportSize(size);
    await app(page);
    await page.locator('details').evaluateAll(nodes => nodes.forEach(n => n.setAttribute('open', '')));
    const dimensions = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight }));
    expect(dimensions.width).toBeLessThanOrEqual(size.width);
    if (size.width > 920) {
      expect(dimensions.height).toBeLessThanOrEqual(size.height);
      const preview = await page.locator('.preview-stage').boundingBox();
      const actions = await page.locator('.persistent-actions').boundingBox();
      await page.locator('.control-tree').evaluate(n => { n.scrollTop = n.scrollHeight; });
      expect(await page.locator('.preview-stage').boundingBox()).toEqual(preview);
      expect(await page.locator('.persistent-actions').boundingBox()).toEqual(actions);
    }
  });
}
