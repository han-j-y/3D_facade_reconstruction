"""Image-only architectural reconstruction of cmp_x0005.jpg.
Run with Blender --background --factory-startup --python build_facade.py.
All dimensions are visual estimates; front is -Y, up is Z. No external textures.
"""
import bpy, math, random, os, sys, json
from mathutils import Vector
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from facade_ornaments import wrought_railing, floral_relief, stone_balustrade
random.seed(17)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for c in list(bpy.data.collections):
    if c.name != 'Collection' and c.users == 0: bpy.data.collections.remove(c)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.length_unit = 'METERS'
scene['Reference'] = 'cmp_x0005.jpg — supplied photograph only'
scene['Method'] = 'Manually interpreted parametric geometry; no other source files or online references.'
scene['Scale'] = 'Estimated 26 m width; all dimensions and unseen depth are approximate.'
scene['Scope'] = 'Detailed front elevation and shallow architectural shell. Rear/interiors are schematic.'
collections = {}
active_col = None
def collection(name):
    global active_col
    if name not in collections:
        c = bpy.data.collections.new(name); scene.collection.children.link(c); collections[name] = c
    active_col = collections[name]
    return active_col

def place(obj):
    active_col.objects.link(obj)
    return obj

def move_new(before):
    for obj in set(bpy.data.objects) - before:
        for col in list(obj.users_collection): col.objects.unlink(obj)
        active_col.objects.link(obj)

def ornament(fn,*args,**kwargs):
    before=set(bpy.data.objects); result=fn(*args,**kwargs); move_new(before); return result

def color(h):
    a=[int(h[i:i+2],16)/255 for i in (0,2,4)]
    return tuple(v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in a)+(1,)

def material(name, hexcolor, rough=.7, metallic=0, noise=0):
    m=bpy.data.materials.new(name); m.diffuse_color=color(hexcolor); m.use_nodes=True
    n=m.node_tree.nodes; p=n.get('Principled BSDF'); p.inputs['Base Color'].default_value=color(hexcolor)
    p.inputs['Roughness'].default_value=rough; p.inputs['Metallic'].default_value=metallic
    if noise:
        tex=n.new('ShaderNodeTexNoise'); tex.inputs['Scale'].default_value=38; tex.inputs['Detail'].default_value=3
        tex.inputs['Roughness'].default_value=.72
        coord=n.new('ShaderNodeTexCoord'); m.node_tree.links.new(coord.outputs['Generated'],tex.inputs['Vector'])
        ramp=n.new('ShaderNodeValToRGB'); base=color(hexcolor)
        ramp.color_ramp.elements[0].position=.12; ramp.color_ramp.elements[1].position=.88
        ramp.color_ramp.elements[0].color=tuple(v*(1-noise) for v in base[:3])+(1,)
        ramp.color_ramp.elements[1].color=tuple(min(1,v*(1+noise)) for v in base[:3])+(1,)
        m.node_tree.links.new(tex.outputs['Fac'],ramp.inputs[0]); m.node_tree.links.new(ramp.outputs['Color'],p.inputs['Base Color'])
        bump=n.new('ShaderNodeBump'); bump.inputs['Strength'].default_value=.14; bump.inputs['Distance'].default_value=.022
        m.node_tree.links.new(tex.outputs['Fac'],bump.inputs['Height']); m.node_tree.links.new(bump.outputs['Normal'],p.inputs['Normal'])
    return m

stone=material('01 | limestone · warm pale carving','BDBBA9',.83,noise=.16)
stone_light=material('02 | dressed limestone edges','CDC9B6',.78,noise=.10)
field=material('03 | weathered gray beige wall field','A6A696',.9,noise=.21)
mortar=material('04 | pale reticulated masonry joints','CECCBA',.94,noise=.08)
plaster=material('05 | smooth upper cream render','C7C8BA',.85,noise=.085)
base_stone=material('06 | ground floor ashlar','AEAEA0',.88,noise=.17)
wood=material('07 | muted petrol green joinery','445B57',.62,noise=.1)
wood_light=material('08 | gray green window mullions','747F74',.54,noise=.06)
iron=material('09 | patinated wrought iron','3A4540',.48,.48,noise=.12)
shadow=material('10 | deep recessed interiors','182622',.98)
glass=material('11 | gray sky glazing','718582',.2,.35)
glass_dark=material('12 | shaded glazing','354642',.24,.2)
glass_light=material('13 | softly reflecting glazing','97A6A1',.26,.22)
curtain=material('14 | warm linen behind glass','AEB1A2',.97,noise=.08)
roof=material('15 | lead gray roof coping','666D68',.62,.35)
ground=material('16 | neutral exhibition ground','D1CFC5',.9)
shop=material('17 | charcoal shop fascia','252D28',.7)
brass=material('18 | aged brass hardware','91836B',.4,.68)

def mesh(name,verts,faces,mat,smooth=False):
    me=bpy.data.meshes.new(name); me.from_pydata(verts,[],faces); me.update()
    ob=place(bpy.data.objects.new(name,me))
    if mat: me.materials.append(mat)
    if smooth:
        for p in me.polygons: p.use_smooth=True
    return ob

def box(name,loc,scale,mat,bevel=0):
    x,y,z=[v/2 for v in scale]
    ob=mesh(name,[(-x,-y,-z),(-x,-y,z),(-x,y,-z),(-x,y,z),(x,-y,-z),(x,-y,z),(x,y,-z),(x,y,z)],[(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3),(0,1,3,2),(4,6,7,5)],mat)
    ob.location=loc
    if bevel:
        mod=ob.modifiers.new('Soft dressed edges','BEVEL'); mod.width=bevel; mod.segments=2
    return ob

def curves(name,paths,r,mat,cyclic=False):
    cu=bpy.data.curves.new(name,'CURVE'); cu.dimensions='3D'; cu.resolution_u=12; cu.bevel_depth=r; cu.bevel_resolution=2
    for path in paths:
        sp=cu.splines.new('POLY'); sp.points.add(len(path)-1)
        for p,co in zip(sp.points,path): p.co=(*co,1)
        sp.use_cyclic_u=cyclic
    ob=place(bpy.data.objects.new(name,cu)); cu.materials.append(mat); return ob

def line(name,path,r,mat,cyclic=False): return curves(name,[path],r,mat,cyclic)

def prism(name,path,z0,z1,mat,bevel=0):
    n=len(path); vs=[(x,y,z) for z in (z0,z1) for x,y in path]
    fs=[tuple(reversed(range(n))),tuple(range(n,n*2))]+[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
    ob=mesh(name,vs,fs,mat)
    if bevel:
        mod=ob.modifiers.new('Rounded stone rim','BEVEL'); mod.width=bevel; mod.segments=3
    return ob

def sphere(name,loc,scale,mat):
    # Latitude/longitude ellipsoid with stable topology.
    verts=[]; seg=12; rings=8
    for j in range(rings+1):
        p=math.pi*j/rings
        for i in range(seg):
            t=2*math.pi*i/seg; verts.append((loc[0]+scale[0]*math.sin(p)*math.cos(t),loc[1]+scale[1]*math.sin(p)*math.sin(t),loc[2]+scale[2]*math.cos(p)))
    faces=[]
    for j in range(rings):
        for i in range(seg): faces.append((j*seg+i,j*seg+(i+1)%seg,(j+1)*seg+(i+1)%seg,(j+1)*seg+i))
    return mesh(name,verts,faces,mat,True)

def cylinder(name,x,y,z0,z1,r,mat):
    return prism(name,[(x+r*math.cos(i*math.tau/16),y+r*math.sin(i*math.tau/16)) for i in range(16)],z0,z1,mat)

def arch_path(x,zspring,rx,rz,y,n=40):
    return [(x+rx*math.cos(math.pi-i*math.pi/n),y,zspring+rz*math.sin(math.pi-i*math.pi/n)) for i in range(n+1)]

def arch_panel(name,x,y,bottom,spring,rx,rz,mat):
    path=[(x-rx,y,bottom),(x+rx,y,bottom)]+list(reversed(arch_path(x,spring,rx,rz,y)))
    return mesh(name,path,[tuple(range(len(path)))],mat)

def moulding(name,x0,x1,z,y=0,depth=.23,h=.17,mat=stone_light):
    box(name+' lower filet',((x0+x1)/2,y-depth/2,z-h*.32),(x1-x0,depth*.85,h*.23),mat,.015)
    box(name+' cyma',((x0+x1)/2,y-depth*.64,z),(x1-x0,depth*1.2,h*.5),mat,.03)
    box(name+' upper lip',((x0+x1)/2,y-depth*.76,z+h*.38),(x1-x0,depth*1.38,h*.22),mat,.014)

def frontframe(name,x,z,w,h,mat=wood,y=-.055,bar=.07):
    box(name+' jamb L',(x-w/2+bar/2,y,z),(bar,.14,h),mat,.012)
    box(name+' jamb R',(x+w/2-bar/2,y,z),(bar,.14,h),mat,.012)
    for zz in (z-h/2+bar/2,z+h/2-bar/2): box(name+' rail',(x,y,zz),(w,.14,bar),mat,.009)

def window(name,x,bottom,w,h,division=2,mat=wood,shutters=False,closed=False,transom=.72):
    z=bottom+h/2
    box(name+' dark reveal',(x,.43,z),(w+.04,.045,h+.04),shadow)
    box(name+' inset jamb L',(x-w/2-.045,.19,z),(.09,.48,h+.15),stone,.018)
    box(name+' inset jamb R',(x+w/2+.045,.19,z),(.09,.48,h+.15),stone,.018)
    box(name+' stone sill',(x,-.10,bottom-.045),(w+.22,.48,.13),stone_light,.025)
    frontframe(name,x,z,w,h,mat)
    if closed:
        box(name+' closed shutter leaf',(x,.065,z),(w-.15,.10,h-.15),wood)
        paths=[]
        for i in range(int(h/.07)):
            zz=bottom+.09+i*.07; paths.append([(x-w/2+.1,-.005,zz),(x+w/2-.1,-.005,zz)])
        curves(name+' shutter louvres',paths,.013,wood_light)
        box(name+' shutter center stile',(x,-.035,z),(.052,.07,h-.13),wood)
    else:
        for j in range(division):
            xx=x-w/2+(j+.5)*w/division
            gm=glass_dark if random.random()<.35 else glass
            box(name+' glass '+str(j),(xx,.16,z),(w/division-.066,.018,h-.13),gm)
            if random.random()<.72:
                cw=w/division*.27
                box(name+' inner linen',(xx-w/division*.22,.137,z-.04),(cw,.018,h-.25),curtain)
        for j in range(1,division): box(name+' vertical mullion',(x-w/2+j*w/division,-.064,z),(.05,.16,h),mat,.008)
        if transom:
            box(name+' transom',(x,-.065,bottom+h*transom),(w,.15,.052),mat,.008)
    if shutters:
        for side in (-1,1): shutter(name+' open shutter',x+side*w/2,bottom,.43,h,side,random.choice([.15,.35,.7]))

def shutter(name,hinge,bottom,w,h,side,angle):
    # One leaf extending from the hinge, tilted toward the street.
    z=bottom+h/2; c=math.cos(angle); s=math.sin(angle)
    center=(hinge+side*w*c/2,-.13-w*s/2,z)
    ob=box(name+' panel',center,(w,.065,h),wood,.012); ob.rotation_euler[2]=-side*angle
    paths=[]
    for i in range(int((h-.1)/.075)):
        zz=bottom+.075+i*.075
        paths.append([(hinge+side*.045*c,-.17-.045*s,zz),(hinge+side*(w-.045)*c,-.17-(w-.045)*s,zz)])
    for a in (.025,w-.025):
        paths.append([(hinge+side*a*c,-.18-a*s,bottom),(hinge+side*a*c,-.18-a*s,bottom+h)])
    curves(name+' fine louvre edges',paths,.012,wood_light)

# STRUCTURE: the front wall is assembled around openings, so reveals are real.
collection('01 · Building shell and level cornices')
box('Simple rear shell — inferred depth',(0,3.15,12.6),(26,.3,25.2),plaster,.05)
for x in (-12.88,12.88): box('Side return — inferred',(x,1.67,12.6),(.36,3.1,25.2),plaster,.02)
levels=[(0,3.65),(3.65,6.80),(6.80,9.85),(9.85,12.90),(12.90,15.95),(15.95,19.00),(19.00,22.05),(22.05,25.25)]
openings={}
openings[0]=[(-10.9,1.85,.20,2.82),(-6.0,2.3,.12,2.84),(0,3.15,0,3.60),(6.05,2.5,.15,2.85),(10.65,1.65,.16,2.85)]
openings[1]=[(-11.15,1.8,4.3,1.95),(-6.05,1.95,4.15,2.05),(-1.53,.70,4.80,1.20),(1.53,.70,4.80,1.20),(6.05,1.95,4.15,2.05),(11.15,1.8,4.3,1.95)]
for k in range(2,6):
    b=levels[k][0]+.51
    openings[k]=[(-6.05,2.10,b,2.24),(-1.61,1.28,b+.11,2.00),(0,1.31,b+.11,2.00),(1.61,1.28,b+.11,2.00),(6.05,2.10,b,2.24)]
for k in (6,7):
    b=levels[k][0]+.63
    openings[k]=[(-6.05,2.26,b,1.65 if k==7 else 1.77),(-1.60,1.23,b+.07,1.38),(0,1.25,b+.07,1.38),(1.60,1.23,b+.07,1.38),(6.05,2.26,b,1.65 if k==7 else 1.77)]

for k,(z0,z1) in enumerate(levels):
    wallmat=base_stone if k==0 else field if k<6 else plaster
    limit=13 if k<2 else 9.32
    last=-limit
    for x,w,b,h in openings[k]:
        left=x-w/2; right=x+w/2
        if left>last: box(f'L{k} solid wall pier',( (last+left)/2,.35,(z0+z1)/2),(left-last,.70,z1-z0),wallmat)
        if b>z0: box(f'L{k} wall below opening',(x,.35,(z0+b)/2),(w,.70,b-z0),wallmat)
        if b+h<z1: box(f'L{k} wall above opening',(x,.35,(b+h+z1)/2),(w,.70,z1-b-h),wallmat)
        last=right
    if last<limit: box(f'L{k} right wall pier',((last+limit)/2,.35,(z0+z1)/2),(limit-last,.70,z1-z0),wallmat)
    if k: box(f'L{k} inset floor plate',(0,1.88,z0-.12),(25.7,2.45,.23),stone)

moulding('Ground floor stringcourse left',-13.08,-1.96,3.68,depth=.35,h=.28)
moulding('Ground floor stringcourse right',1.96,13.08,3.68,depth=.35,h=.28)
moulding('Transition to upper smooth render',-9.35,9.35,19.0,depth=.27,h=.26)
moulding('Roof entablature',-9.35,9.35,25.19,depth=.38,h=.33)
box('Roof parapet',(0,.26,25.58),(18.7,.52,.60),stone_light,.025)
moulding('Parapet cap',-9.38,9.38,25.88,depth=.27,h=.13)
for z in (6.73,9.80,12.85,15.90):
    box('Fine horizontal masonry course',(0,-.027,z),(18.64,.10,.045),stone_light,.008)

# Characteristic irregular round-ended pale masonry outlines.
collection('02 · Reticulated masonry relief')
def is_wall(x,z,pad=.045):
    k=next((i for i,(a,b) in enumerate(levels) if a<=z<b),None)
    if k is None or k==0 or k>5: return False
    limit=12.98 if k==1 else 9.30
    if abs(x)>limit: return False
    for xx,w,b,h in openings[k]:
        if xx-w/2-pad<x<xx+w/2+pad and b-pad<z<b+h+pad: return False
    # Central limestone ornament covers this region.
    if abs(x)<2.66: return False
    return True

paths=[]
row=0; z=3.91
while z<18.90:
    h=.29; step=.38; width=.90
    offset=.45 if row%2 else 0
    for j in range(-16,16):
        cx=j*width+offset; ww=width*.90+random.uniform(-.035,.035)
        p=[]
        for n in range(40):
            t=n*math.tau/40
            # Superellipse forms organic rounded ashlar rather than a square grid.
            co=math.cos(t); si=math.sin(t)
            xx=cx+ww/2*math.copysign(abs(co)**.42,co)+.018*math.sin(3*t+row)
            zz=z+h/2*math.copysign(abs(si)**.72,si)+.014*math.cos(4*t+j)
            p.append((xx,.002,zz))
        good=[is_wall(a,c) for a,bb,c in p]
        if all(good): paths.append(p+[p[0]])
        else:
            run=[]
            for point,valid in zip(p+p[:1],good+good[:1]):
                if valid: run.append(point)
                else:
                    if len(run)>2: paths.append(run)
                    run=[]
            if len(run)>2: paths.append(run)
    row+=1; z+=step
curves('Hand interpreted rounded masonry outlines',paths,.016,mortar)
# Sparse dressed ashlar joints at street level.
paths=[]
for z in (.48,1.0,1.52,2.04,2.56,3.08):
    for a,b in [(-13,-11.83),(-9.97,-7.16),(-4.83,-1.75),(1.75,4.79),(7.31,9.79),(11.51,13)]: paths.append([(a,-.013,z),(b,-.013,z)])
curves('Ground ashlar horizontal joints',paths,.012,stone)

# SIDE BALCONY STACKS AND ALL FLAT WINDOWS.
collection('03 · Windows and blue green shutters')
for k in range(1,8):
    for j,(x,w,b,h) in enumerate(openings[k]):
        if k==1 and abs(x)<2: continue
        middle=abs(x)<2.5
        window(f'L{k} window {j+1}',x,b,w,h,2,wood_light if k>=6 else wood,
               shutters=(abs(abs(x)-6.05)<.1 and k<7),closed=(middle and abs(x)<.2 and k<6),transom=.78 if k<6 else .56)

collection('04 · Four organic balconies each side')
def balcony_path(cx,w=3.34,depth=1.05):
    pts=[]
    # Swept soft shoulders, slightly concave center: characteristic organic slab.
    for i in range(41):
        t=math.pi-i*math.pi/40
        x=cx+w/2*math.cos(t)
        y=-.11-depth*math.sin(t)**.40+.12*math.exp(-((x-cx)/.40)**2)
        pts.append((x,y))
    return pts
for k in range(2,6):
    for side in (-1,1):
        x=side*6.05; z=levels[k][0]+.32; p=balcony_path(x)
        prism(f'L{k} organic balcony slab',p,z-.13,z+.07,stone,.055)
        p2=[(x+(xx-x)*1.025,yy-.02) for xx,yy in p]
        prism(f'L{k} balcony upper rolled edge',p2,z+.04,z+.13,stone_light,.035)
        line(f'L{k} organic lower bead',[(a,b,z-.14) for a,b in p],.048,stone_light)
        for sx in (-1,1):
            bx=x+sx*.95
            line('Swept limestone balcony corbel',[(bx,-.14,z-.79),(bx,-.26,z-.59),(bx,-.62,z-.29),(bx,-.91,z-.17)],.13,stone)
            sphere('Corbel volute',(bx,-.36,z-.44),(.18,.14,.22),stone)
        ornament(wrought_railing,f'L{k} ornate balcony {side}',[(a,b-.01) for a,b in p],z+.16,.92,iron,ornate=True)

collection('05 · Rectangular upper balconies')
for k in (6,7):
    for side in (-1,1):
        x=side*6.05; z=levels[k][0]+.42; w=3.30
        box(f'L{k} cantilever rectangular balcony',(x,-.58,z),(w,1.35,.23),stone,.04)
        box(f'L{k} balcony deep fascia',(x,-1.23,z-.08),(w,.16,.30),stone,.025)
        p=[(x-w/2,-.06),(x-w/2,-1.22),(x+w/2,-1.22),(x+w/2,-.06)]
        ornament(wrought_railing,f'L{k} simple upper railing {side}',p,z+.14,.94,wood_light,ornate=False)

# SIX GLAZED TIERS AT EACH CURVED CORNER.
collection('06 · Rounded corner bays — six tiers')
def bay_arc(cx,rx=1.84,ry=1.40,n=56,y=.12):
    return [(cx+rx*math.cos(math.pi-i*math.pi/n),y-ry*math.sin(math.pi-i*math.pi/n)) for i in range(n+1)]

for side in (-1,1):
    cx=side*11.15
    # Rounded lower springing ledge and corbel under glazed stack.
    for z,rx,ry in [(6.48,1.54,1.04),(6.63,1.72,1.26),(6.79,1.94,1.52)]:
        prism('Rounded bay springing course',bay_arc(cx,rx,ry),z-.11,z+.08,stone,.04)
    for k in range(2,8):
        z0,z1=levels[k]; bottom=z0+.23; top=z1-.22
        arc=bay_arc(cx,1.83,1.38,n=8)
        prism(f'Corner {side} L{k} dark interior',bay_arc(cx,1.72,1.25),bottom,top,shadow)
        # Eight faceted glass panels describe the bowed window geometry.
        for i in range(8):
            x1,y1=arc[i]; x2,y2=arc[i+1]
            vec=Vector((x2-x1,y2-y1,0)); length=vec.length
            mx=(x1+x2)/2; my=(y1+y2)/2; ang=math.atan2(y2-y1,x2-x1)
            gm=([glass_light,glass,glass_dark,glass][(i+k+side)%4])
            ob=box(f'Corner {side} L{k} glazed facet {i}',(mx,my,(bottom+top)/2),(length-.055,.035,top-bottom-.05),gm,.007); ob.rotation_euler[2]=ang
            # Some lowered interior blinds, partly interrupted to preserve glazed rhythm.
            if (i+k)%4==0 and k<6:
                ob=box('Linen shade within curved bay',(mx,my-.022,bottom+(top-bottom)*.28),(length*.82,.015,(top-bottom)*.48),curtain); ob.rotation_euler[2]=ang
            for frac in ((.44,.83) if k>=6 else (.76,)):
                zz=bottom+(top-bottom)*frac
                ob=box('Curved bay transom',(mx,my-.018,zz),(length+.04,.08,.055),wood_light if k>=6 else wood,.006); ob.rotation_euler[2]=ang
        for i,(x,y) in enumerate(arc):
            box('Corner bay vertical glazing mullion',(x,y-.02,(bottom+top)/2),(.066,.095,top-bottom+.04),wood_light if k>=6 else wood,.012)
        # Solid creamy floor bands, rolled front edges.
        for zz,thick,rx,ry in [(z0+.07,.24,1.94,1.51),(z1-.12,.23,1.94,1.51)]:
            p=bay_arc(cx,rx,ry)
            prism('Continuous curved limestone bay band',p,zz-thick/2,zz+thick/2,stone_light,.04)
        if k<6:
            p=bay_arc(cx,2.00,1.68)
            prism('Corner rounded balcony shelf',p,z0+.13,z0+.26,stone,.04)
            ornament(wrought_railing,f'Corner {side} L{k} curved scroll ironwork',p,z0+.28,.90,iron,ornate=True)
            # Upper delicate scalloped iron fringe echoing photo fan-shaped transoms.
            paths=[]
            for i in range(8):
                a=arc[i]; b=arc[i+1]
                paths.append([(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t-.045,top-.12-.14*math.sin(math.pi*t)) for t in [j/14 for j in range(15)]])
            curves('Bay upper fan tracery',paths,.017,wood_light)
    p=bay_arc(cx,1.96,1.54)
    prism('Corner rounded roof parapet',p,25.25,25.72,stone,.04)
    prism('Corner broad coping',bay_arc(cx,2.01,1.59),25.70,25.86,stone_light,.035)
    ornament(wrought_railing,'Corner roof fine safety railing',bay_arc(cx,1.75,1.35),25.88,.63,iron,ornate=False)

# THE ORNATE FOUR-STOREY CENTRAL TRIPLE WINDOW FEATURE.
collection('07 · Central limestone pilasters and carved spandrels')
for k in range(2,6):
    z0,z1=levels[k]; b=z0+.62
    # Three deeply framed window surrounds, four continuous slim columns.
    for j,x in enumerate((-1.61,0,1.61)):
        box('Central carved spandrel backing',(x,-.19,z0+.14),(1.53,.36,.71),stone,.10)
        ornament(floral_relief,f'L{k} carved swag panel {j}',x,-.41,z0+.17,1.34,.58,stone_light)
        box('Central window sill roll',(x,-.30,b-.10),(1.46,.43,.14),stone_light,.045)
        # Arched eyebrow follows subtle upward central sweep.
        path=[(x-.75+1.50*t,-.28,b+2.02+.13*math.sin(math.pi*t)) for t in [i/32 for i in range(33)]]
        line('Soft arched window eyebrow',path,.068,stone_light)
    for x in (-2.42,-.805,.805,2.42):
        box('Narrow central pilaster shaft',(x,-.24,z0+1.57),(.155,.30,2.22),stone_light,.038)
        for zz in (z0+.44,z0+2.69):
            box('Pilaster collar',(x,-.26,zz),(.24,.34,.10),stone,.025)
            sphere('Carved pilaster bulb',(x,-.27,zz+.11),(.14,.20,.17),stone_light)
        line('Pilaster vertical raised bead',[(x,-.409,z0+.63),(x,-.409,z0+2.52)],.023,stone_light)
    moulding('Central horizontal carved sill',-2.62,2.62,z0-.20,y=-.1,depth=.32,h=.17)

# Crown with curling stone eyebrows and floral frieze medallions.
path=[(-2.66,-.19,18.67),(-2.54,-.20,18.85),(-2.19,-.21,18.97),(-1.55,-.23,19.035),(0,-.24,19.08),(1.55,-.23,19.035),(2.19,-.21,18.97),(2.54,-.20,18.85),(2.66,-.19,18.67)]
line('Central crest large curved pediment',path,.073,stone_light)
for x in (-8.39,-3.72,3.72,8.39):
    ornament(floral_relief,'Frieze floral arabesque',x,-.145,18.58,1.24,.51,stone_light)

# Broad stone balustrades above the shops.
collection('08 · First-floor terraces and stone balusters')
for side in (-1,1):
    if side<0: a,b=-13.05,-4.35
    else: a,b=4.35,13.05
    p=[(a,-.05),(a,-.82),(b,-.82),(b,-.05)]
    box('Broad terrace slab',((a+b)/2,-.42,3.91),(b-a,1.18,.26),stone_light,.035)
    ornament(stone_balustrade,'Stone terrace balustrade',p,4.06,.83,stone_light)
    for x in (a,b): box('Stone terrace end pier',(x,-.82,4.47),(.28,.32,.88),stone,.035)

# Ornamental first-floor central landing and carved entrance.
collection('09 · Sculpted portal and entrance iron gates')
box('Portal central first floor stone field',(0,-.13,5.35),(5.0,.28,2.91),stone,.04)
# Carve the entrance's rounded head through the actual wall and portal field.
# The rectangular doorway was already left open during wall assembly.
profile=[(-1.435,-.05),(1.435,-.05)]+[(p[0],p[2]) for p in reversed(arch_path(0,2.72,1.435,1.415,0))]
nv=len(profile)
verts=[(x,y,z) for y in (-.95,.91) for x,z in profile]
faces=[tuple(range(nv-1,-1,-1)),tuple(range(nv,nv*2))]+[(i,(i+1)%nv,(i+1)%nv+nv,i+nv) for i in range(nv)]
cutter=mesh('Temporary arched doorway cutter',verts,faces,None)
for ob in list(bpy.data.objects):
    if ob.type!='MESH' or ob==cutter: continue
    if not (ob.name.startswith('L1 ') or ob.name.startswith('Portal central first floor stone field')): continue
    lo=ob.location.x-ob.dimensions.x/2; hi=ob.location.x+ob.dimensions.x/2
    if hi < -1.44 or lo > 1.44: continue
    mod=ob.modifiers.new('True arched entrance opening','BOOLEAN'); mod.operation='DIFFERENCE'; mod.solver='EXACT'; mod.object=cutter
    bpy.context.view_layer.objects.active=ob
    bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.data.objects.remove(cutter,do_unlink=True)
for x in (-1.56,1.56):
    # Slender polygonal light with clipped, Art Nouveau shoulders.
    coords=[(x-.37,-.305,4.91),(x-.37,-.305,5.85),(x-.20,-.305,5.98),(x-.13,-.305,6.15),(x+.14,-.305,6.15),(x+.23,-.305,6.00),(x+.37,-.305,5.91),(x+.37,-.305,4.91),(x+.20,-.305,4.77),(x-.20,-.305,4.77)]
    mesh('Irregular lower central window',coords,[tuple(range(len(coords)))],shadow)
    line('Sculpted polygonal window surround',coords,.088,stone_light,True)
    line('Lower window inset frame',[(a,-.329,c) for a,b,c in coords],.029,wood,True)
    line('Small window transom',[(x-.30,-.35,5.30),(x+.30,-.35,5.30)],.025,wood_light)
    for dx in (-.52,.52): line('Flowing portal vertical relief',[(x+dx,-.36,4.2),(x+dx,-.38,5.3),(x+dx,-.35,6.35)],.037,stone_light)
for j in range(3):
    # Almond cartouche aligned over the center entrance.
    w=.48-j*.12; h=.76-j*.12
    p=[(w*math.sin(t)*abs(math.sin(t))**.4,-.40-j*.035,5.30+h*math.cos(t)) for t in [i*math.tau/64 for i in range(64)]]
    line('Central almond cartouche',p,.035,stone_light,True)
ornament(floral_relief,'Leaf inside almond cartouche',0,-.54,5.30,.43,.91,stone_light)

arch_panel('Dark arched entrance recess',0,.20,.03,2.72,1.43,1.41,shadow)
for r in (1.48,1.67,1.86):
    path=[(-r,-.38,.04),(-r,-.38,2.68)]+arch_path(0,2.68,r,r*.90,-.38)[1:]+[(r,-.38,.04)]
    line('Portal concentric molded arch',path,.085,stone_light)
for side in (-1,1):
    x=side*1.85
    box('Portal base plinth',(x,-.20,.39),(.49,.57,.75),stone,.038)
    # Spiral at the shoulder of the doorway.
    path=[]
    for i in range(101):
        t=i/100*math.pi*4; r=.42*(1-i/120)
        path.append((side*.98+side*r*math.cos(t),-.42,4.17+r*math.sin(t)))
    line('Portal large carved spiral',path,.065,stone_light)
    ornament(floral_relief,'Portal vertical botanical stem',x,-.35,2.1,.40,2.70,stone_light)

# Dark paired entry doors and fanlight grille.
for side in (-1,1):
    x=side*.665
    frontframe('Main double gate',x,1.42,1.29,2.75,wood,y=-.02,bar=.09)
    for i in range(8):
        xx=x-.51+i*.145
        line('Entry gate vertical iron bar',[(xx,-.13,.20),(xx,-.13,2.61)],.016,iron)
    for zz in (.58,2.18,2.51): line('Entry gate horizontal rail',[(x-.56,-.145,zz),(x+.56,-.145,zz)],.023,iron)
    for zz in (.89,1.68):
        for sx in (-1,1):
            p=[(x+sx*.22*math.sin(t),-.16,zz+.32*math.cos(t)) for t in [i*math.tau/40 for i in range(41)]]
            line('Gate oval scroll',p,.014,iron)
    box('Gate brass handle',(side*.13,-.22,1.31),(.035,.05,.25),brass,.012)
line('Fanlight lower beam',[(-1.35,-.07,2.74),(1.35,-.07,2.74)],.045,wood)
line('Fanlight arched frame',arch_path(0,2.74,1.34,1.28,-.07),.044,wood)
for i in range(1,12):
    xx=-1.34+i*2.68/12; top=2.74+1.28*math.sqrt(max(0,1-(xx/1.34)**2))
    line('Fanlight radial grille',[(xx*.7,-.12,2.77),(xx,-.12,top)],.018,iron)

# Street-floor glazing; photo-obscured signage left deliberately unlettered.
collection('10 · Ground-floor shopfronts')
for j,(x,w,b,h) in enumerate(openings[0]):
    if abs(x)<.1: continue
    window(f'Shopfront {j}',x,b,w,h,division=2,mat=shop,transom=.74)
    box('Unlettered shop fascia',(x,-.10,b+h-.19),(w,.17,.36),shop,.025)
    box('Shop threshold',(x,-.12,.11),(w+.15,.50,.14),stone_light,.025)
    for s in (-1,1):
        xx=x+s*(w/2+.10)
        box('Dressed shop jamb',(xx,-.06,b+h/2),(.16,.27,h+.10),stone,.025)
    # Curved shallow lintel over each storefront.
    line('Shop shallow arched lintel',[(x-w/2-.14,-.12,b+h+.10),(x-w/2+.06,-.13,b+h+.26),(x+w/2-.06,-.13,b+h+.26),(x+w/2+.14,-.12,b+h+.10)],.055,stone_light)

# ROOF TERRACE AND SET-BACK PENTHOUSE.
collection('11 · Roof terrace, railing and penthouse')
box('Roof surface',(0,1.7,25.30),(25.7,3.3,.20),roof,.025)
for a,b in [(-8.95,-4.9),(2.65,8.95)]:
    ornament(wrought_railing,'Roof terrace railing',[(a,.09),(b,.09)],25.95,.66,iron,ornate=False)
box('Set-back rooftop volume',(-1.05,1.44,26.24),(8.1,2.25,1.63),roof,.035)
for i in range(6):
    xx=-4.91+(i+.5)*7.72/6
    box('Penthouse glazed panel',(xx,.286,26.36),(7.72/6-.10,.055,1.16),glass_dark,.01)
    box('Penthouse white vertical frame',(xx-.60,.228,26.36),(.055,.10,1.26),wood_light,.008)
box('Penthouse top overhang',(-1.05,1.35,27.12),(8.47,2.58,.16),roof,.035)
box('Penthouse front eave',(-1.05,.03,27.08),(8.47,.13,.19),stone,.018)
for x in (-8.9,8.9):
    line('Roof small drain pipe',[(x,-.41,24.97),(x,-.31,25.57),(x,.31,25.75)],.049,iron)

# PRESENTATION. Two cameras; no photographic street occluders.
collection('12 · Presentation — cameras, lighting and ground')
box('Architectural plinth',(0,.9,-.21),(28.0,7.1,.40),stone,.08)
box('Ground',(0,0,-.48),(2000,2000,.12),ground,.04)

def camera(name,loc,target,ortho):
    data=bpy.data.cameras.new(name); ob=place(bpy.data.objects.new(name,data)); ob.location=loc
    ob.rotation_euler=(Vector(target)-ob.location).to_track_quat('-Z','Y').to_euler(); data.type='ORTHO'; data.ortho_scale=ortho; data.lens=50
    return ob
cam=camera('Camera · front elevation',(0,-58,14.0),(0,0,13.7),31.15)
hero=camera('Camera · three quarter',(-34,-62,32),(0,0,13.0),33.4)
detail=camera('Camera · carved central facade',(-11,-33,19),(0,-.3,13.2),18)
scene.camera=hero

world=bpy.data.worlds.new('Soft overcast architectural sky'); world.use_nodes=True
world.node_tree.nodes.get('Background').inputs[0].default_value=(.72,.79,.83,1)
world.node_tree.nodes.get('Background').inputs[1].default_value=.40; scene.world=world
def area(name,loc,power,size,col,target):
    d=bpy.data.lights.new(name,'AREA'); d.energy=power; d.shape='DISK'; d.size=size; d.color=col
    o=place(bpy.data.objects.new(name,d)); o.location=loc; o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
area('Large softbox · upper left',(-14,-21,34),4500,16,(1.0,.93,.82),(0,0,13))
area('Cool frontal sky fill',(15,-18,20),2800,13,(.84,.93,1),(0,0,13))
d=bpy.data.lights.new('Gentle raking daylight','SUN'); d.energy=1.7; d.angle=math.radians(18)
o=place(bpy.data.objects.new(d.name,d)); o.rotation_euler=(math.radians(28),math.radians(-24),math.radians(-30))
scene.render.engine='CYCLES'; scene.cycles.samples=32; scene.cycles.use_denoising=True
scene.cycles.max_bounces=6; scene.cycles.diffuse_bounces=3; scene.cycles.glossy_bounces=3
scene.render.resolution_x=1400; scene.render.resolution_y=1500; scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'; scene.render.film_transparent=False
scene.view_settings.view_transform='AgX'
scene.render.image_settings.color_mode='RGB'
scene.render.fps=24
# Pack only the supplied photograph for convenient inspection inside Blender.
reference_path='C:/Users/hiroo/3D_facade_reconstruction/data/baseForBatch/cmp_x0005.jpg'
reference=bpy.data.images.load(reference_path,check_existing=True)
reference.name='REFERENCE · supplied cmp_x0005.jpg'; reference.pack(); reference.use_fake_user=True
bg=cam.data.background_images.new(); bg.image=reference; bg.alpha=.35
cam.data.show_background_images=False
# Set a usable starting view in the saved Blender file.
for screen in bpy.data.screens:
    for ar in screen.areas:
        if ar.type=='VIEW_3D':
            ar.spaces.active.region_3d.view_perspective='CAMERA'
            ar.spaces.active.clip_end=1000
            ar.spaces.active.shading.type='MATERIAL'

readme=bpy.data.texts.new('READ ME — image-only reconstruction')
readme.write('CMP X0005 | IMAGE-ONLY FACADE STUDY\n\n'
 'Source: user-supplied cmp_x0005.jpg only.\n'
 '26 m overall width; 27.2 m rooftop height. Dimensions and depth are estimates.\n'
 'Ground floor + seven residential levels; six-tier curved corner bays.\n'
 'All windows, balconies, balusters, ironwork, masonry outlines and reliefs are editable geometry.\n'
 'Collections are organized by architectural system. Front faces negative Y; Z is up.\n'
 'Three cameras: front elevation, three quarter, central detail.\n'
 'The unobserved sides and rear form a shallow schematic shell; interiors are not reconstructed.\n'
 'Street cars, people, temporary signs, wires and lamppost are excluded to expose the architecture.\n'
 'No external texture dependencies. The scene is fully self-contained.\n')
blend=OUT/'cmp_x0005_facade.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(blend))
stats={'objects':len(bpy.data.objects),'meshes':len(bpy.data.meshes),'curves':len(bpy.data.curves),'materials':len(bpy.data.materials),'width_estimate_m':26,'rooftop_height_estimate_m':27.2,'source':'supplied cmp_x0005.jpg only'}
(OUT/'model_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print('FACADE_BUILD_COMPLETE',json.dumps(stats),flush=True)
scene.render.resolution_x=1050; scene.render.resolution_y=1150
scene.render.filepath=str(OUT/'facade_three_quarter.png')
bpy.ops.render.render(write_still=True)
scene.camera=cam; scene.render.resolution_x=1300; scene.render.resolution_y=1450
scene.render.filepath=str(OUT/'facade_front.png')
bpy.ops.render.render(write_still=True)
scene.camera=hero; scene.render.resolution_x=1400; scene.render.resolution_y=1500
scene.render.filepath=str(OUT/'facade_three_quarter.png')
bpy.ops.wm.save_as_mainfile(filepath=str(blend))
print('ALL_RENDERS_COMPLETE',flush=True)
