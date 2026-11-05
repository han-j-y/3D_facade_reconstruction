"""Validate only this task's generated Blender artifact."""
import bpy, json
from pathlib import Path

scene=bpy.context.scene
assert len(bpy.data.objects)>1500, 'Missing facade geometry'
assert len(bpy.data.cameras)==3, 'Missing presentation camera'
assert len([c for c in bpy.data.collections if ' · ' in c.name])==12
assert scene.camera is not None
reference=bpy.data.images.get('REFERENCE · supplied cmp_x0005.jpg')
assert reference is not None and reference.packed_file is not None, 'Reference is not packed'
assert len([o for o in bpy.data.objects if o.type=='MESH' and 'glazed facet' in o.name])==96
assert not any(o.type=='MESH' and len(o.data.vertices)==0 for o in bpy.data.objects), 'Empty geometry'
assert not any(m.type=='BOOLEAN' and m.object is None for o in bpy.data.objects for m in o.modifiers), 'Unresolved cutout'
assert not any(i.source=='FILE' and not i.packed_file and i.users for i in bpy.data.images), 'External image dependency'
report={'status':'PASS','saved_file':bpy.data.filepath,'objects':len(bpy.data.objects),'collections':12,'cameras':3,'curved_bay_glazed_panels':96,'reference_packed':True,'external_images_required':False}
Path(__file__).with_name('validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('FACADE_VALIDATION',json.dumps(report),flush=True)
