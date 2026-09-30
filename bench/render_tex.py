"""pyrender 로 텍스처 GLB 를 4방향 정사영 렌더 (조명 flat 에 가깝게)."""
import os; os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np, trimesh, pyrender
from PIL import Image
def render_sheet(glb, out, size=512):
    m = trimesh.load(glb, force='mesh')
    m.apply_translation(-m.bounds.mean(0)); m.apply_scale(1.0 / np.abs(m.vertices).max())
    tiles = []
    for az in (0, 90, 180, 270):
        mm = m.copy(); mm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(az), [0, 1, 0]))
        sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.9, 0.9, 0.9])
        sc.add(pyrender.Mesh.from_trimesh(mm))
        cam = pyrender.OrthographicCamera(xmag=1.15, ymag=1.15)
        sc.add(cam, pose=trimesh.transformations.translation_matrix([0, 0, 5]))
        sc.add(pyrender.DirectionalLight(intensity=1.5), pose=trimesh.transformations.translation_matrix([0, 0, 5]))
        r = pyrender.OffscreenRenderer(size, size); col, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
        im = Image.fromarray(col, 'RGBA'); bg = Image.new('RGBA', im.size, (255, 255, 255, 255)); tiles.append(Image.alpha_composite(bg, im).convert('RGB'))
    W = Image.new('RGB', (size * 4, size), 'white')
    for i, t in enumerate(tiles): W.paste(t, (i * size, 0))
    W.save(out); print('->', out)
if __name__ == '__main__':
    import sys; render_sheet(sys.argv[1], sys.argv[2])
