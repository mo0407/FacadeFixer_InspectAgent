"""Deterministic synthetic demonstration, not facade ground truth."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .image_inspection import ImageInspectionRequest, run_image_inspection
from .model_gateway import ModelConfig
from .playbook import Playbook, Rule
from .storage import ArtifactStore, sha256


class SyntheticExperts:
    """Known fixture boxes; deliberately not an inference model."""

    def __init__(self):
        self.calls = []

    async def call(self, name, arguments):
        if name != 'facade_models_detect':
            raise ValueError('Unexpected demo tool')
        self.calls.append({'name':name, 'arguments':arguments})
        path = Path(arguments['image_paths'][0])
        with Image.open(path) as image:
            width,height = image.size
        local = path.name.startswith('local-')
        boxes = [] if local else [
            {'instance_id':'1','class_code':'C','class_name':'Crack','bbox_xyxy':[280,200,325,385],'confidence':.86},
            {'instance_id':'2','class_code':'P','class_name':'Peeling','bbox_xyxy':[400,430,485,495],'confidence':.34},
        ]
        # Verify both low-confidence Ghost anchors and full-image duplicate merging.
        return {'status':'completed','coordinate_space':'original_pixels',
            'images':[{'source_sha256':sha256(path),'width':width,'height':height,
                       'defects':boxes,'unmapped_defects':[]}],
            'source':{'method':'synthetic_fixture','model_profile':arguments['model_profile']}}


async def run_image_demo(output_dir: Path, store: ArtifactStore):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    image = Image.new('RGB',(900,660),'#d8dde6')
    draw = ImageDraw.Draw(image)
    draw.rectangle((150,40,750,620),fill='#ead7bd',outline='#a79683',width=3)
    for y in (80,240,400):
        for x in (180,530,650):
            draw.rectangle((x,y,x+65,y+100),fill='#597c94',outline='#334e68',width=3)
    draw.line([(294,205),(303,245),(290,286),(314,332),(307,380)],fill='#423d3b',width=4)
    draw.polygon([(405,438),(456,432),(482,460),(469,492),(419,486)],fill='#8d8173')
    image.save(output_dir/'wall.png')
    mask = Image.new('L',image.size,0)
    scope_draw = ImageDraw.Draw(mask)
    scope_draw.rectangle((150,40,750,620),fill=255)
    for y in (80,240,400):
        for x in (180,530,650):
            scope_draw.rectangle((x,y,x+65,y+100),fill=0)
    mask.save(output_dir/'wall-scope.png')
    request = ImageInspectionRequest(image_path=str(output_dir/'wall.png'),scope_mask_path=str(output_dir/'wall-scope.png'))
    settings = ModelConfig.model_validate({'schema_version':'model-gateway-v1','api_enabled':False,
        'connections':{'fixture':{'adapter':'mock','model':'synthetic-fixture','supports_images':True}},
        'roles':{role:'fixture' for role in ('generator','reflector','curator')}})
    playbook = Playbook(version='P0-demo',rules=[Rule(id='wall-only',condition='All inspections',
        recommendation='Inspect only primary-building wall surfaces; ignore windows and background.',
        limitations='Synthetic example; not an empirically validated rule.')])
    store.write_json(output_dir/'request.json',request.model_dump())
    store.write_json(output_dir/'models.json',settings.model_dump())
    store.write_json(output_dir/'playbook.json',playbook.model_dump())
    experts = SyntheticExperts()
    result = await run_image_inspection(request,store,settings,playbook,experts=experts)
    folder = Path(result['report_path']).parent
    store.write_json(folder/'synthetic_tool_calls.json',experts.calls)
    return result
