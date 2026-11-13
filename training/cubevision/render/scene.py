"""
The Blender scene that draws a ``Scenario`` and measures what the camera sees.

Built once (cube, hands, lights, camera, backdrop), then each image only moves
objects and changes material values, which keeps rendering fast. Requires
``bpy`` (Blender as a Python module, or a script run by the Blender binary).

Frames: the cube's own axes follow StickerCube (x R, y U, z F); Blender's
world has z up. The cuber stands at -y (Blender) and looks along +y: the
near face of the cube is at -y, the far face at +y, the right face at +x.
"""
from __future__ import annotations  # noqa: I001 (bpy must be imported before bmesh)

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# bpy first: with the bpy module, bmesh and bpy_extras exist once bpy is imported
import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

from ..cube import N, POSITIONS, cubie_of, parse_move
from . import hand_mesh
from .sampler import Finger, Hand, Scenario

#: StickerCube axes (x R, y U, z towards the cuber) → Blender (x right, y away from the cuber, z up)
TS_TO_BL = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
HALF = 1.5  # half the cube's side: cubies are 1 unit
#: Rough projected size of the cube, to frame it
EXTENT = 3.75
#: Outward normal of each face (StickerCube axes), faces U R F D L B
#: Points of a tile tested for visibility (shares of its side, from its centre): the 5 inner ones, then 8 near its edges
TILE_POINTS = ((0, 0), (0.28, 0.28), (-0.28, 0.28), (0.28, -0.28), (-0.28, -0.28),
               (0.42, 0), (-0.42, 0), (0, 0.42), (0, -0.42), (0.42, 0.42), (-0.42, 0.42), (0.42, -0.42), (-0.42, -0.42))
FACE_NORMALS = ((0, 1, 0), (1, 0, 0), (0, 0, 1), (0, -1, 0), (-1, 0, 0), (0, 0, -1))


@dataclass
class Rendered:
    """What one render measured, for the labels."""
    visible: list[bool]  # per tile (StickerCube order)
    facing: list[float]  # per face: cosine between its normal and the direction to the camera
    #: The whole cube's box (hidden parts included), x0 y0 x1 y1 as shares of the image (y down),
    #: not clipped to the image; None when the cube is not drawn
    box: list[float] | None
    #: Rotation (row-major 3×3) taking the WCA directions of the centres (white +y, red +x,
    #: green +z) to where those centres point in the camera's frame (x right, y up, z towards
    #: the viewer); None when the cube is not drawn
    orientation: list[float] | None
    #: Per tile: its projected area, share of the image's area (0 when facing away)
    area: list[float] | None = None
    #: Per tile: how many of its 5 inner points the camera sees (see _measure)
    inner: list[int] | None = None


def centre_frame(stickers: list[int]) -> Matrix:
    """Rotation (StickerCube axes) taking each centre colour's WCA direction to where that centre is."""
    m = Matrix(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    for face in range(6):
        colour = stickers[face * 9 + 4] // 9
        n, c = FACE_NORMALS[face], FACE_NORMALS[colour]
        for i in range(3):
            for j in range(3):
                m[i][j] += n[i] * c[j] / 2
    return m


def to_bl(v) -> Vector:
    return TS_TO_BL @ Vector(v)


def quat_to_bl(q: list[float]) -> Quaternion:
    """A rotation given in StickerCube axes, in Blender axes (TS_TO_BL is a proper rotation)."""
    w, x, y, z = q
    return Quaternion((w, *(TS_TO_BL @ Vector((x, y, z)))))


def srgb_to_linear(c: list[float]) -> tuple[float, float, float, float]:
    lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return (lin[0], lin[1], lin[2], 1.0)


def kelvin_rgb(kelvin: float) -> tuple[float, float, float]:
    """Approximate colour of a black body (Tanner Helland's fit), 0..1."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    return tuple(min(1.0, max(0.0, c / 255.0)) for c in (r, g, b))  # type: ignore[return-value]


def _principled(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    if hasattr(mat, "use_nodes") and not mat.node_tree:
        mat.use_nodes = True
    return mat


def _bsdf(mat: bpy.types.Material):
    return mat.node_tree.nodes["Principled BSDF"]


def _set(node, name: str, value) -> None:
    """A socket's value, left alone when unchanged: any write makes Cycles compile the material again (a
    video's frames change only the cube's and hands' places)."""
    socket = node.inputs.get(name)
    if socket is None:
        return
    old = socket.default_value
    same = abs(old - value) < 1e-6 if isinstance(old, float | int) else all(abs(a - b) < 1e-6 for a, b in zip(old, value, strict=False))
    if not same:
        socket.default_value = value


def _smooth(mesh: bpy.types.Mesh) -> None:
    for poly in mesh.polygons:
        poly.use_smooth = True


def rounded_box(name: str, size: float, radius: float) -> bpy.types.Mesh:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bmesh.ops.bevel(bm, geom=bm.verts[:] + bm.edges[:], offset=radius, segments=3, profile=0.5, affect="EDGES")
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    _smooth(mesh)
    return mesh


def tile_mesh(name: str, radius: float, thickness: float) -> bpy.types.Mesh:
    """A unit square tile (±0.5) with rounded corners, ``thickness`` along z, centred on the origin."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.z *= thickness
    vertical = [e for e in bm.edges if abs(e.verts[0].co.x - e.verts[1].co.x) < 1e-6 and abs(e.verts[0].co.y - e.verts[1].co.y) < 1e-6]
    bmesh.ops.bevel(bm, geom=vertical + list({v for e in vertical for v in e.verts}), offset=radius, segments=4, profile=0.5, affect="EDGES")
    top = [e for e in bm.edges if all(v.co.z > thickness * 0.49 for v in e.verts)]
    bmesh.ops.bevel(bm, geom=top + list({v for e in top for v in e.verts}), offset=thickness * 0.35, segments=2, profile=0.5, affect="EDGES")
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    _smooth(mesh)
    return mesh


def disc_mesh() -> bpy.types.Mesh:
    """A unit disc (diameter 1, height 1) centred on the origin."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=40, radius1=0.5, radius2=0.5, depth=1.0)
    mesh = bpy.data.meshes.new("disc")
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def cylinder_mesh() -> bpy.types.Mesh:
    """Radius 1, from z -1 to 1, open ends (hidden inside the knuckles' spheres)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False, segments=20, radius1=1.0, radius2=1.0, depth=2.0)
    mesh = bpy.data.meshes.new("cylinder")
    bm.to_mesh(mesh)
    bm.free()
    _smooth(mesh)
    return mesh


def ellipsoid_mesh() -> bpy.types.Mesh:
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=12, radius=1.0)
    mesh = bpy.data.meshes.new("ellipsoid")
    bm.to_mesh(mesh)
    bm.free()
    _smooth(mesh)
    return mesh


class CubeScene:
    """One Blender scene reused for every image of a worker."""

    # Thickness of a sticker and of a stickerless tile, in cube units
    THICKNESS = {"stickered": 0.012, "stickered_white": 0.012, "stickerless": 0.05}

    def __init__(self, resolution: int = 192, samples: int = 32, device: str = "auto", backgrounds: list[Path] | None = None):
        """``resolution``: side of the square crops, or longer side of the full frames."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.resolution = resolution
        import addon_utils

        addon_utils.enable("cycles", default_set=True)
        self.scene = bpy.context.scene
        self.backgrounds = backgrounds or []
        self.device = self._setup_render(resolution, samples, device)

        self.root = bpy.data.objects.new("CubeRoot", None)
        self.root.rotation_mode = "QUATERNION"
        self._link(self.root)

        self.body_meshes = [rounded_box(f"body{i}", 0.98, r) for i, r in enumerate((0.05, 0.09, 0.13))]
        self.tile_meshes = {
            style: [tile_mesh(f"tile-{style}-{i}", r, t) for i, r in enumerate((0.1, 0.18, 0.26))]
            for style, t in self.THICKNESS.items()
        }
        self.ellipsoid = ellipsoid_mesh()
        self.disc = disc_mesh()
        self.cylinder = cylinder_mesh()
        # Meshes are shared between objects; each object links its own material to the one slot
        for mesh in [*self.body_meshes, *(m for ms in self.tile_meshes.values() for m in ms), self.ellipsoid, self.disc, self.cylinder]:
            mesh.materials.append(None)

        self.body_material = _principled("body")
        self.bodies: dict[tuple[int, int, int], bpy.types.Object] = {}
        for p in POSITIONS:
            c = cubie_of(p)
            if c in self.bodies:
                continue
            body = self._object(f"cubie{c}", self.body_meshes[0], self.body_material)
            body.parent = self.root
            body["rest"] = list(to_bl(c) / 2)
            self.bodies[c] = body

        self.tiles: list[bpy.types.Object] = []
        self.tile_materials: list[bpy.types.Material] = []
        for i, p in enumerate(POSITIONS):
            c = cubie_of(p)
            normal = to_bl(tuple(a - b for a, b in zip(p, c, strict=True)))
            mat = self._tile_material(f"tile{i:02d}")
            tile = self._object(f"tile{i:02d}", self.tile_meshes["stickered"][0], mat)
            tile.parent = self.bodies[c]
            tile.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(normal)
            tile["normal"] = list(normal)
            self.tiles.append(tile)
            self.tile_materials.append(mat)
        # A cap on each centre tile, shown per image (Scenario.caps)
        self.caps = {i: self._object(f"cap{i:02d}", self.disc, _principled(f"cap{i:02d}")) for i in range(4, 54, 9)}
        for i, cap in self.caps.items():
            cap.parent = self.tiles[i]

        self.skin_materials = [_principled(f"skin{i}") for i in range(2)]
        # A palm and 5 fingers of 3 segments each, posed per image
        self.hands = [[self._object(f"hand{i}-{k}", self.ellipsoid, self.skin_materials[i]) for k in range(16)] for i in range(2)]
        # Hands of Hand.model "capsules": per finger three phalanges, four knuckles (base to tip), a nail
        self.nail_materials = [_principled(f"nail{i}") for i in range(2)]
        self.phalanges = [[self._object(f"phalanx{i}-{k}", self.cylinder, self.skin_materials[i]) for k in range(15)] for i in range(2)]
        self.knuckles = [[self._object(f"knuckle{i}-{k}", self.ellipsoid, self.skin_materials[i]) for k in range(20)] for i in range(2)]
        self.nails = [[self._object(f"nail{i}-{k}", self.ellipsoid, self.nail_materials[i]) for k in range(5)] for i in range(2)]
        # Hands of Hand.model "mesh": a sculpted hand (hand_mesh), its vertices moved per image
        self.rig = hand_mesh.Rig.load() if hand_mesh.ASSET.exists() else None
        self.mesh_hands = [self._mesh_hand(i) for i in range(2)] if self.rig else []
        self._wrists: list = [None, None]
        # The cuber around the cube (Scenario.body)
        self.forearms = [self._object(f"forearm{i}", self.ellipsoid, _principled(f"forearm{i}")) for i in range(2)]
        self.torso = self._object("torso", self.ellipsoid, _principled("torso"))
        self.head = self._object("head", self.ellipsoid, _principled("head"))
        self.occluder_materials = [_principled(f"occluder{i}") for i in range(3)]
        self.occluders = [self._object(f"occluder{i}", self.ellipsoid, m) for i, m in enumerate(self.occluder_materials)]

        # Things around the cube that are not it (Scenario.distractors), each with a grid-of-colours material
        self.distractor_materials = [self._grid_material(f"distractor{i}") for i in range(4)]
        self.distractors = [self._object(f"distractor{i}", self.body_meshes[1], m) for i, m in enumerate(self.distractor_materials)]

        self.backdrop = self._plane("backdrop")
        self.table = self._plane("table")
        self.lights = []
        for i in range(3):
            data = bpy.data.lights.new(f"light{i}", "AREA")
            light = bpy.data.objects.new(f"light{i}", data)
            self._link(light)
            self.lights.append(light)
        cam_data = bpy.data.cameras.new("camera")
        self.camera = bpy.data.objects.new("camera", cam_data)
        self._link(self.camera)
        self.scene.camera = self.camera
        self.scene.world = bpy.data.worlds.new("world")
        if hasattr(self.scene.world, "use_nodes") and not self.scene.world.node_tree:
            self.scene.world.use_nodes = True

    # ─── Setup ──────────────────────────────────────────────────────────────

    def _link(self, obj: bpy.types.Object) -> None:
        self.scene.collection.objects.link(obj)

    def _object(self, name: str, mesh: bpy.types.Mesh, material: bpy.types.Material) -> bpy.types.Object:
        obj = bpy.data.objects.new(name, mesh)
        obj.rotation_mode = "QUATERNION"
        self._link(obj)
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = material
        return obj

    def _plane(self, name: str) -> bpy.types.Object:
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        mesh.materials.append(_principled(name))
        obj = bpy.data.objects.new(name, mesh)
        obj.rotation_mode = "QUATERNION"
        self._link(obj)
        return obj

    def _setup_render(self, resolution: int, samples: int, device: str) -> str:
        s = self.scene
        s.render.engine = "CYCLES"
        s.render.resolution_x = s.render.resolution_y = resolution
        s.render.resolution_percentage = 100
        s.render.image_settings.file_format = "JPEG"
        s.render.image_settings.quality = 95
        s.cycles.samples = samples
        s.cycles.use_adaptive_sampling = True
        s.cycles.use_denoising = True
        s.cycles.max_bounces = 4
        s.cycles.transparent_max_bounces = 2
        s.render.film_transparent = False
        # Cycles keeps the scene between renders and only updates what changed: 3 times the images
        # per second, same pixels (tests/test_render.py renders A, B, then A again)
        s.render.use_persistent_data = True
        chosen = "CPU"
        if device != "cpu":
            prefs = bpy.context.preferences.addons["cycles"].preferences
            kinds = ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"] if device == "auto" else [device.upper()]
            for kind in kinds:
                try:
                    prefs.compute_device_type = kind
                except TypeError:
                    continue
                prefs.refresh_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    s.cycles.device = "GPU"
                    # The same denoiser (OpenImageDenoise) on the GPU: 1.5 to 1.7 times the images per second
                    s.cycles.denoising_use_gpu = True
                    chosen = kind
                    break
            if chosen == "CPU" and device != "auto":
                raise RuntimeError(f"no {device} device found")
        return chosen

    @staticmethod
    def _tile_material(name: str) -> bpy.types.Material:
        """A tile's plastic: its colour (an RGB node) times a printed pattern (noise), the pattern's weight 0 for plain tiles."""
        mat = _principled(name)
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        colour = nodes.new("ShaderNodeRGB")
        colour.name = "colour"
        coords = nodes.new("ShaderNodeTexCoord")
        noise = nodes.new("ShaderNodeTexNoise")
        noise.name = "pattern"
        noise.inputs["Detail"].default_value = 6.0
        links.new(coords.outputs["Object"], noise.inputs["Vector"])
        shade = nodes.new("ShaderNodeMapRange")
        shade.inputs["To Min"].default_value = 0.55
        shade.inputs["To Max"].default_value = 1.3
        links.new(noise.outputs["Fac"], shade.inputs["Value"])
        mix = nodes.new("ShaderNodeMix")
        mix.name = "print"
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 0.0
        links.new(colour.outputs["Color"], mix.inputs["A"])
        links.new(shade.outputs["Result"], mix.inputs["B"])
        links.new(mix.outputs["Result"], _bsdf(mat).inputs["Base Color"])
        return mat

    def _mesh_hand(self, i: int) -> bpy.types.Object:
        """A sculpted hand, the right one mirrored for the left (i 0): smooth, subdivided once at render,
        its rest place and nails as attributes for the skin's material."""
        rig = self.rig
        mesh = bpy.data.meshes.new(f"realhand{i}")
        faces = rig.faces if i else rig.faces[:, ::-1]  # mirrored: the faces' winding reversed
        mesh.from_pydata([tuple(v) for v in rig.verts], [], [tuple(int(x) for x in f) for f in faces])
        rest = mesh.attributes.new("rest", "FLOAT_VECTOR", "POINT")
        rest.data.foreach_set("vector", (rig.verts * hand_mesh.UNIT).ravel())
        nail = mesh.attributes.new("nail", "FLOAT", "POINT")
        nail.data.foreach_set("value", rig.nail)
        _smooth(mesh)
        obj = bpy.data.objects.new(f"realhand{i}", mesh)
        self._link(obj)
        sub = obj.modifiers.new("smooth", "SUBSURF")
        sub.levels, sub.render_levels = 0, 1
        obj.data.materials.append(self._skin_material(f"realskin{i}"))
        obj.hide_render = obj.hide_viewport = True
        return obj

    @staticmethod
    def _skin_material(name: str) -> bpy.types.Material:
        """Skin: blotches of colour and fine grain (on the rest place, so they stay on the skin as it
        moves), light under the surface, nails paler and glossier."""
        mat = _principled(name)
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = _bsdf(mat)
        rest = nodes.new("ShaderNodeAttribute")
        rest.attribute_name = "rest"
        nail = nodes.new("ShaderNodeAttribute")
        nail.attribute_name = "nail"
        skin = nodes.new("ShaderNodeRGB")
        skin.name = "skin"
        blotch = nodes.new("ShaderNodeTexNoise")
        blotch.inputs["Scale"].default_value = 1.2
        blotch.inputs["Detail"].default_value = 4.0
        links.new(rest.outputs["Vector"], blotch.inputs["Vector"])
        # Skin a little redder and darker here and there
        tint = nodes.new("ShaderNodeMix")
        tint.data_type = "RGBA"
        tint.blend_type = "MULTIPLY"
        socket = lambda sockets, identifier: next(x for x in sockets if x.identifier == identifier)  # noqa: E731
        links.new(blotch.outputs["Fac"], socket(tint.inputs, "Factor_Float"))
        links.new(skin.outputs["Color"], socket(tint.inputs, "A_Color"))
        socket(tint.inputs, "B_Color").default_value = (0.82, 0.62, 0.58, 1.0)
        nail_colour = nodes.new("ShaderNodeMix")
        nail_colour.data_type = "RGBA"
        nail_colour.blend_type = "SCREEN"
        links.new(nail.outputs["Fac"], socket(nail_colour.inputs, "Factor_Float"))
        links.new(socket(tint.outputs, "Result_Color"), socket(nail_colour.inputs, "A_Color"))
        socket(nail_colour.inputs, "B_Color").default_value = (0.55, 0.42, 0.40, 1.0)
        links.new(socket(nail_colour.outputs, "Result_Color"), bsdf.inputs["Base Color"])
        rough = nodes.new("ShaderNodeMapRange")
        links.new(nail.outputs["Fac"], rough.inputs["Value"])
        rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.5, 0.18
        links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
        # Pores and creases: a fine bump
        grain = nodes.new("ShaderNodeTexNoise")
        grain.inputs["Scale"].default_value = 28.0
        grain.inputs["Detail"].default_value = 6.0
        links.new(rest.outputs["Vector"], grain.inputs["Vector"])
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.12
        bump.inputs["Distance"].default_value = 0.02
        links.new(grain.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        _set(bsdf, "Subsurface Weight", 0.3)
        _set(bsdf, "Subsurface Radius", Vector((1.0, 0.35, 0.2)))
        _set(bsdf, "Subsurface Scale", 0.08)
        _set(bsdf, "Specular IOR Level", 0.45)
        return mat

    def _pose_mesh_hand(self, i: int, hand: Hand, tricks: list[dict], seed: int) -> None:
        """The sculpted hand holding the cube (hand_mesh.hold), mirrored for the left."""
        obj = self.mesh_hands[i]
        left = hand.side < 0
        if left:
            tricks = [hand_mesh.mirror_trick(t) for t in tricks]
        pose = hand_mesh.hold(self.rig, hand, tricks, seed)
        verts = self.rig.skin(pose.transforms)
        wrist, forearm = pose.wrist, pose.forearm
        if left:
            verts = verts * np.array([-1.0, 1.0, 1.0])
            wrist, forearm = wrist * np.array([-1.0, 1.0, 1.0]), forearm * np.array([-1.0, 1.0, 1.0])
        obj.data.vertices.foreach_set("co", verts.astype(np.float32).ravel())
        obj.data.update()
        self._wrists[i] = (Vector(wrist), Vector(forearm))
        rgb = obj.data.materials[0].node_tree.nodes["skin"].outputs["Color"]
        colour = srgb_to_linear(hand.skin)
        if any(abs(a - b) > 1e-6 for a, b in zip(rgb.default_value, colour, strict=False)):
            rgb.default_value = colour

    # ─── One image ──────────────────────────────────────────────────────────

    def render(self, sc: Scenario, path: Path | None, measure: bool = True) -> Rendered | None:
        """Draw the scenario to ``path`` (JPEG) and measure what the labels need (unless not ``measure``:
        a video's frames have their truth); without ``path``, only measure (relabel.py)."""
        self._resolution(sc)
        self._cube(sc)
        self._hands(sc)
        self._occluders(sc)
        self._motion(sc)
        self.scene.frame_set(1)
        cam_dir = self._camera(sc)
        self._body(sc, cam_dir)
        self._backdrop(sc, cam_dir)
        self._distractors(sc, cam_dir)
        self._lights(sc, cam_dir)
        self.scene.frame_set(1)
        if not measure:
            rendered = None
        elif sc.cube_hidden:
            rendered = Rendered([False] * 54, [0.0] * 6, None, None)
        else:
            visible, facing, area, inner = self._measure()
            rendered = Rendered(visible, facing, self._box(), self._orientation(sc), area, inner)
        if path is not None:
            self.scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
        return rendered

    def _resolution(self, sc: Scenario) -> None:
        r = self.scene.render
        if sc.aspect >= 1:
            r.resolution_x, r.resolution_y = self.resolution, max(2, round(self.resolution / sc.aspect / 2) * 2)
        else:
            r.resolution_x, r.resolution_y = max(2, round(self.resolution * sc.aspect / 2) * 2), self.resolution

    def _cube(self, sc: Scenario) -> None:
        self.root.rotation_quaternion = quat_to_bl(sc.orientation)
        for obj in [*self.bodies.values(), *self.tiles]:
            obj.hide_render = obj.hide_viewport = sc.cube_hidden
        body_mesh = self.body_meshes[sc.bevel]
        bsdf = _bsdf(self.body_material)
        _set(bsdf, "Base Color", srgb_to_linear(sc.body_color))
        _set(bsdf, "Roughness", sc.roughness + 0.15)
        for body in self.bodies.values():
            body.data = body_mesh
            body.location = Vector(body["rest"])
            body.rotation_quaternion = Quaternion()
        thickness = self._thickness = self.THICKNESS[sc.style]
        mesh = self.tile_meshes[sc.style][sc.bevel]
        for i, tile in enumerate(self.tiles):
            tile.data = mesh
            tile.scale = (sc.sticker_size, sc.sticker_size, 1.0)
            tile.location = Vector(tile["normal"]) * (0.49 + thickness / 2)
            color = [min(1.0, c * sc.tile_jitter[i]) for c in sc.palette[sc.stickers[i] // 9]]
            nodes = self.tile_materials[i].node_tree.nodes
            rgb = nodes["colour"].outputs["Color"]
            if any(abs(a - b) > 1e-6 for a, b in zip(rgb.default_value, srgb_to_linear(color), strict=True)):
                rgb.default_value = srgb_to_linear(color)
            _set(nodes["print"], "Factor", sc.texture["amount"] if sc.texture else 0.0)
            if sc.texture:
                _set(nodes["pattern"], "Scale", sc.texture["scale"])
            t = _bsdf(self.tile_materials[i])
            _set(t, "Roughness", sc.roughness)
            _set(t, "Coat Weight", sc.coat if sc.coat is not None else 0.3 if sc.style == "stickerless" else 0.0)
        self._set_caps(sc, thickness)

    def _set_caps(self, sc: Scenario, thickness: float) -> None:
        caps = sc.caps
        for i, cap in self.caps.items():
            colour = sc.stickers[i] // 9
            logo = caps is not None and caps["logo"] == colour
            shown = caps is not None and (logo or caps["all"])
            # Out of the depsgraph too, so that hidden caps never stop a visibility ray
            cap.hide_render = cap.hide_viewport = not shown or sc.cube_hidden
            if not shown:
                continue
            cap.scale = (caps["size"], caps["size"], caps["height"])
            cap.location = Vector((0, 0, thickness / 2 + caps["height"] / 2))
            base = caps["logo_color"] if logo else [min(1.0, c * caps["shade"]) for c in sc.palette[colour]]
            m = _bsdf(cap.material_slots[0].material)
            _set(m, "Base Color", srgb_to_linear(base))
            _set(m, "Roughness", min(1.0, sc.roughness * caps["gloss"]))

    def _turning_bodies(self, sc: Scenario, turn=None, exclude=()) -> tuple[list[bpy.types.Object], Vector] | None:
        turn = turn or sc.turn
        if not turn:
            return None
        move = parse_move(turn.move)
        assert move is not None
        layers = move.layers()
        axis_ts = [0, 0, 0]
        axis_ts[move.axis] = 1
        bodies = [b for c, b in self.bodies.items() if c[move.axis] in layers and b not in exclude]
        return bodies, to_bl(axis_ts)

    def _next_bodies(self, sc: Scenario, first) -> tuple[list[bpy.types.Object], Vector] | None:
        """The layer of a corner cut, without the cubies the first layer still carries."""
        if not sc.turn_next:
            return None
        return self._turning_bodies(sc, sc.turn_next, exclude=set(first[0]) if first else ())

    def _slack(self, sc: Scenario, *moving) -> None:
        """The layers a few degrees off (sc.slack), but the cubies of those turning."""
        if not sc.slack:
            return
        taken = {b for m in moving if m for b in m[0]}
        axis = sc.slack["axis"]
        axis_ts = [0, 0, 0]
        axis_ts[axis] = 1
        for layer, degrees in zip((-(N - 1), 0, N - 1), sc.slack["angles"], strict=True):
            self._set_turn([b for c, b in self.bodies.items() if c[axis] == layer and b not in taken], to_bl(axis_ts), degrees)

    def _set_turn(self, bodies: list[bpy.types.Object], axis: Vector, degrees: float) -> None:
        q = Quaternion(axis, math.radians(degrees))
        for body in bodies:
            body.location = q @ Vector(body["rest"])
            body.rotation_quaternion = q

    def _hands(self, sc: Scenario) -> None:
        for i, parts in enumerate(self.hands):
            hand = sc.hands[i] if i < len(sc.hands) else None
            capsules = hand is not None and hand.model == "capsules"
            sculpted = hand is not None and hand.model == "mesh" and self.rig is not None
            for obj in parts:
                obj.hide_render = obj.hide_viewport = hand is None or sculpted
            for obj in [*self.phalanges[i], *self.knuckles[i], *self.nails[i]]:
                obj.hide_render = obj.hide_viewport = not capsules
            if self.mesh_hands:
                self.mesh_hands[i].hide_render = self.mesh_hands[i].hide_viewport = not sculpted
            self._wrists[i] = None
            if hand is None:
                continue
            if sculpted:
                self._pose_mesh_hand(i, hand, [t for t in sc.tricks if t["side"] == hand.side], sc.index * 2 + i)
                continue
            bsdf = _bsdf(self.skin_materials[i])
            _set(bsdf, "Base Color", srgb_to_linear(hand.skin))
            _set(bsdf, "Roughness", 0.5)
            _set(bsdf, "Subsurface Weight", 0.25)
            _set(bsdf, "Subsurface Radius", Vector((1.0, 0.4, 0.25)))
            _set(bsdf, "Subsurface Scale", 0.15)
            if capsules:
                nail = _bsdf(self.nail_materials[i])
                _set(nail, "Base Color", srgb_to_linear([min(1.0, 0.25 + 0.85 * c) for c in hand.skin]))
                _set(nail, "Roughness", 0.25)
                self._pose_capsule_hand(i, parts[0], hand, [t for t in sc.tricks if t["side"] == hand.side])
            else:
                self._pose_hand(parts, hand)

    def _pose_hand(self, parts: list[bpy.types.Object], hand: Hand) -> None:
        s, r, h = hand.side, hand.finger_radius, HALF
        palm = parts[0]
        palm.location = Vector((s * (h + 0.32 * hand.palm_scale + 0.12), hand.palm_depth, hand.palm_height))
        palm.rotation_quaternion = Quaternion()
        palm.scale = Vector((0.32, 1.25, 1.35)) * hand.palm_scale
        chains: list[tuple[Finger, str]] = [(hand.thumb, "near")] + [(f, f.target) for f in hand.fingers]
        k = 1
        for finger, target in chains:
            path = self._finger_path(hand, finger, target)
            for a, b in zip(path, path[1:], strict=False):
                seg = parts[k]
                seg.hide_render = seg.hide_viewport = False
                d = b - a
                seg.location = (a + b) / 2
                seg.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
                seg.scale = (r, r, d.length / 2 + r * 0.55)
                k += 1
        for seg in parts[k:]:
            seg.hide_render = seg.hide_viewport = True

    @staticmethod
    def _trick_rotation(trick: dict) -> Quaternion:
        return Quaternion(to_bl(trick["axis"]), math.radians(trick["angle"]))

    def _pose_capsule_hand(self, i: int, palm: bpy.types.Object, hand: Hand, tricks: list[dict] = ()) -> None:
        """Palm as the blobs', fingers as round phalanges between knuckles, tapering to the tip, a nail on it.
        ``tricks``: this hand's pushes (Scenario.tricks): a finger laid on the turning layer and carried
        round with it, or the whole hand turned with it (a wrist turn)."""
        s, h = hand.side, HALF
        wrist = next((t for t in tricks if t["finger"] < 0), None)
        whole = self._trick_rotation({**wrist, "angle": wrist["angle"] * wrist["weight"]}) if wrist else Quaternion()
        palm.location = whole @ Vector((s * (h + 0.3 * hand.palm_scale + 0.12), hand.palm_depth, hand.palm_height))
        palm.rotation_quaternion = whole
        palm.scale = Vector((0.3, 1.15, 1.25)) * hand.palm_scale
        for obj in self.hands[i][1:]:
            obj.hide_render = obj.hide_viewport = True
        outward = {"far": Vector((0, 1, 0)), "near": Vector((0, -1, 0)), "top": Vector((0, 0, 1))}
        chains = [(hand.thumb, "near", 1.12)] + [(f, f.target, 1.0 - 0.06 * k) for k, f in enumerate(hand.fingers)]
        for f, (finger, target, girth) in enumerate(chains):
            path = [whole @ v for v in self._finger_path(hand, finger, target)]
            out = whole @ outward[target]
            push = next((t for t in tricks if t["finger"] == f), None)
            if push:
                # Laid on the layer (blended in as the turn comes), then carried round with it
                laid = self._finger_path(hand, Finger(push["target"], push["across"], push["reach"]), push["target"])
                q, w = self._trick_rotation(push), push["weight"]
                path = [a.lerp(q @ b, w) for a, b in zip(path, laid, strict=True)]
                out = out.lerp(q @ outward[push["target"]], w).normalized()
            radii = [hand.finger_radius * girth * t for t in (1.0, 0.95, 0.88, 0.8)]
            for k, (p, r) in enumerate(zip(path, radii, strict=True)):
                knuckle = self.knuckles[i][f * 4 + k]
                knuckle.location, knuckle.rotation_quaternion, knuckle.scale = p, Quaternion(), (r, r, r)
            for k, (a, b) in enumerate(zip(path, path[1:], strict=False)):
                d = b - a
                seg = self.phalanges[i][f * 3 + k]
                seg.location = (a + b) / 2
                seg.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
                r = (radii[k] + radii[k + 1]) / 2
                seg.scale = (r, r, d.length / 2)
            # The nail: on the tip's back, away from the cube, along the last phalanx
            tip, last, r = path[-1], (path[-1] - path[-2]).normalized(), radii[-1]
            nail = self.nails[i][f]
            nail.location = tip + out * r * 0.8 - last * r * 0.35
            nail.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(out)
            nail.scale = (r * 0.7, r * 0.7, r * 0.18)
        for f in range(len(chains), 5):
            for obj in [*self.knuckles[i][f * 4:f * 4 + 4], *self.phalanges[i][f * 3:f * 3 + 3], self.nails[i][f]]:
                obj.hide_render = obj.hide_viewport = True

    @staticmethod
    def _finger_path(hand: Hand, finger: Finger, target: str) -> list[Vector]:
        """Points of a finger from the palm round an edge of the cube onto a face (cuber's frame)."""
        s, r, h, a = hand.side, hand.finger_radius, HALF, finger.across
        side_x = s * (h + r + 0.12)
        corner_x = s * (h + r * 0.7)
        def on_face(depth: float) -> float:
            return s * (h - depth)
        if target == "far":
            return [Vector((side_x, hand.palm_depth + 0.9 * hand.palm_scale, a)), Vector((corner_x, h + r * 0.7, a)),
                    Vector((on_face(finger.reach * 0.5), h + r, a)), Vector((on_face(finger.reach), h + r, a))]
        if target == "near":
            return [Vector((side_x, hand.palm_depth - 0.9 * hand.palm_scale, a)), Vector((corner_x, -h - r * 0.7, a)),
                    Vector((on_face(finger.reach * 0.5), -h - r, a)), Vector((on_face(finger.reach), -h - r, a))]
        return [Vector((side_x, a, hand.palm_height + hand.palm_scale)), Vector((corner_x, a, h + r * 0.7)),
                Vector((on_face(finger.reach * 0.5), a, h + r)), Vector((on_face(finger.reach), a, h + r))]

    def _occluders(self, sc: Scenario) -> None:
        for i, obj in enumerate(self.occluders):
            blob = sc.occluders[i] if i < len(sc.occluders) else None
            obj.hide_render = obj.hide_viewport = blob is None
            if blob is None:
                continue
            obj.location = to_bl(blob["position"])
            obj.scale = blob["size"]
            _set(_bsdf(self.occluder_materials[i]), "Base Color", srgb_to_linear(blob["color"]))

    def _camera(self, sc: Scenario) -> Vector:
        """Place the camera; a crop is framed on the cube's measured box, as the app crops it."""
        az, el = math.radians(sc.azimuth), math.radians(sc.elevation)
        direction = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
        distance = sc.distance
        data = self.camera.data
        data.lens_unit = "FOV"
        data.clip_start = 0.05
        data.clip_end = 500
        data.shift_x = data.shift_y = 0.0
        location = direction * distance
        self.camera.location = location
        self.camera.rotation_mode = "QUATERNION"
        fraction = sc.frame_fraction if sc.framing == "full" else 0.6
        half_fov = math.atan((EXTENT / 2) / (fraction * distance))
        data.angle = 2 * half_fov
        target = Vector((0, 0, 0))
        if sc.framing == "full":
            # Anywhere in the frame: aim beside the cube (offsets as shares of the width and height)
            look = (-direction).to_track_quat("-Z", "Y")
            longer = 2 * distance * math.tan(half_fov)
            width, height = (longer, longer / sc.aspect) if sc.aspect >= 1 else (longer * sc.aspect, longer)
            target = (look @ Vector((1, 0, 0))) * (sc.offset[0] * width) + (look @ Vector((0, 1, 0))) * (-sc.offset[1] * height)
        data.dof.use_dof = sc.focus is not None
        if sc.focus is not None:
            data.dof.focus_distance = sc.focus * distance
            data.dof.aperture_fstop = sc.fstop
        view = (target - location).normalized()
        self.camera.rotation_quaternion = Quaternion(view, math.radians(sc.roll)) @ view.to_track_quat("-Z", "Y")
        if sc.framing == "crop" and not sc.cube_hidden:
            # Zoom so that the box's longer side times the margin fills the image, then shift the
            # lens to centre the box (plus the detector's error): the app's crop, exactly
            self.scene.frame_set(1)
            x0, y0, x1, y1 = self._box()
            k = max(x1 - x0, y1 - y0) * sc.crop_margin
            data.angle = 2 * math.atan(math.tan(half_fov) * k)
            cx, cy = 0.5 + ((x0 + x1) / 2 - 0.5) / k, 0.5 + ((y0 + y1) / 2 - 0.5) / k
            data.shift_x = (cx - 0.5) + sc.offset[0]
            data.shift_y = -((cy - 0.5) + sc.offset[1])
        return direction

    def _box(self) -> list[float]:
        """The cube's box in the image (all its pieces, hidden or not), shares of the width and height, y down."""
        depsgraph = bpy.context.evaluated_depsgraph_get()
        xs, ys = [], []
        for body in self.bodies.values():
            mw = body.evaluated_get(depsgraph).matrix_world
            for corner in body.bound_box:
                x, y, _z = world_to_camera_view(self.scene, self.camera, mw @ Vector(corner))
                xs.append(x)
                ys.append(1 - y)
        return [min(xs), min(ys), max(xs), max(ys)]

    def _orientation(self, sc: Scenario) -> list[float]:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cam_from_world = self.camera.evaluated_get(depsgraph).matrix_world.to_3x3().normalized().transposed()
        world_from_cube = self.root.evaluated_get(depsgraph).matrix_world.to_3x3().normalized() @ TS_TO_BL
        m = cam_from_world @ world_from_cube @ centre_frame(sc.stickers)
        return [round(m[i][j], 5) for i in range(3) for j in range(3)]

    def _body(self, sc: Scenario, cam_dir: Vector) -> None:
        body = sc.body
        parts = [*self.forearms, self.torso, self.head]
        if body is None:
            for obj in parts:
                obj.hide_render = obj.hide_viewport = True
            return
        clothing = srgb_to_linear(body["clothing"])
        for i, forearm in enumerate(self.forearms):
            hand = sc.hands[i] if i < len(sc.hands) else None
            forearm.hide_render = forearm.hide_viewport = hand is None
            if hand is None:
                continue
            spec, s, ps = body["forearms"][i], hand.side, hand.palm_scale
            wrist = Vector((s * (HALF + 0.32 * ps + 0.12), hand.palm_depth - 0.9 * ps, hand.palm_height - 0.5))
            # Towards the cuber (-y) and down, a little outwards
            d = Vector((s * spec["spread"], -1.0, -spec["drop"])).normalized()
            if self._wrists[i]:
                # A sculpted hand's own forearm, the sleeve or the arm going on from it
                wrist, d = self._wrists[i]
                wrist = wrist + d * 0.6
            forearm.location = wrist + d * (spec["length"] / 2 - 0.3)
            forearm.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
            forearm.scale = (spec["radius"], spec["radius"] * 0.85, spec["length"] / 2)
            m = _bsdf(forearm.material_slots[0].material)
            _set(m, "Base Color", clothing if body["sleeves"] else srgb_to_linear(hand.skin))
            _set(m, "Roughness", 0.8 if body["sleeves"] else 0.5)
        # Torso and head behind the cube, unless the camera films from the cuber's side
        shown = cam_dir.y > -0.3
        for obj in (self.torso, self.head):
            obj.hide_render = obj.hide_viewport = not shown
        if not shown:
            return
        self.torso.location = Vector((0, -body["chest"] - 5, -body["height"]))
        self.torso.rotation_quaternion = Quaternion()
        self.torso.scale = (9.0, 5.0, 14.0)
        _set(_bsdf(self.torso.material_slots[0].material), "Base Color", clothing)
        _set(_bsdf(self.torso.material_slots[0].material), "Roughness", 0.85)
        self.head.location = Vector((0, -body["chest"] - 5, -body["height"] + 20))
        self.head.rotation_quaternion = Quaternion()
        self.head.scale = (5.0, 5.5, 6.5)
        _set(_bsdf(self.head.material_slots[0].material), "Base Color", srgb_to_linear(body["skin"]))

    @staticmethod
    def _grid_material(name: str) -> bpy.types.Material:
        """Cells of random colours (one of six) with dark gaps between them, on any box: built once, set per image."""
        mat = _principled(name)
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = _bsdf(mat)
        coords = nodes.new("ShaderNodeTexCoord")
        scale = nodes.new("ShaderNodeMapping")
        scale.name = "cells"
        links.new(coords.outputs["Generated"], scale.inputs["Vector"])
        cell = nodes.new("ShaderNodeVectorMath")
        cell.operation = "FLOOR"
        links.new(scale.outputs["Vector"], cell.inputs[0])
        noise = nodes.new("ShaderNodeTexWhiteNoise")
        noise.name = "noise"
        noise.noise_dimensions = "4D"
        links.new(cell.outputs["Vector"], noise.inputs["Vector"])
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.name = "palette"
        ramp.color_ramp.interpolation = "CONSTANT"
        for k in range(1, 6):
            ramp.color_ramp.elements.new(k / 6)
        ramp.color_ramp.elements.remove(ramp.color_ramp.elements[-1])  # the default one at 1: six in all
        links.new(noise.outputs["Value"], ramp.inputs["Fac"])
        # A gap where two of the three coordinates are near a cell's edge (one always is, on a face)
        frac = nodes.new("ShaderNodeVectorMath")
        frac.operation = "FRACTION"
        links.new(scale.outputs["Vector"], frac.inputs[0])
        split = nodes.new("ShaderNodeSeparateXYZ")
        links.new(frac.outputs["Vector"], split.inputs[0])
        total = None
        for axis in ("X", "Y", "Z"):
            centred = nodes.new("ShaderNodeMath")
            centred.operation = "SUBTRACT"
            centred.inputs[1].default_value = 0.5
            links.new(split.outputs[axis], centred.inputs[0])
            far = nodes.new("ShaderNodeMath")
            far.operation = "ABSOLUTE"
            links.new(centred.outputs[0], far.inputs[0])
            edge = nodes.new("ShaderNodeMath")
            edge.operation = "GREATER_THAN"
            edge.name = f"edge{axis}"
            links.new(far.outputs[0], edge.inputs[0])
            if total is None:
                total = edge
            else:
                add = nodes.new("ShaderNodeMath")
                add.operation = "ADD"
                links.new(total.outputs[0], add.inputs[0])
                links.new(edge.outputs[0], add.inputs[1])
                total = add
        gap = nodes.new("ShaderNodeMath")
        gap.operation = "GREATER_THAN"
        gap.inputs[1].default_value = 1.5
        links.new(total.outputs[0], gap.inputs[0])
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.name = "gap"
        socket = lambda sockets, identifier: next(x for x in sockets if x.identifier == identifier)  # noqa: E731
        links.new(gap.outputs[0], socket(mix.inputs, "Factor_Float"))
        links.new(ramp.outputs["Color"], socket(mix.inputs, "A_Color"))
        socket(mix.inputs, "B_Color").default_value = (0.01, 0.01, 0.01, 1.0)
        links.new(socket(mix.outputs, "Result_Color"), bsdf.inputs["Base Color"])
        return mat

    def _distractors(self, sc: Scenario, cam_dir: Vector) -> None:
        """Place each one in the view: behind the cube or beside it, across the view's width and height."""
        cam = self.camera
        right = cam.rotation_quaternion @ Vector((1, 0, 0))
        up = cam.rotation_quaternion @ Vector((0, 1, 0))
        half = math.tan(cam.data.angle / 2)
        width, height = (1.0, 1 / sc.aspect) if sc.aspect >= 1 else (sc.aspect, 1.0)
        cam_distance = cam.location.length
        for i, obj in enumerate(self.distractors):
            spec = sc.distractors[i] if i < len(sc.distractors) else None
            obj.hide_render = obj.hide_viewport = spec is None
            if spec is None:
                continue
            depth = min(spec["depth"], sc.background["distance"] - 1.0)
            reach = half * (cam_distance + depth)
            obj.location = -cam_dir * depth + right * (spec["across"][0] * reach * width) + up * (spec["across"][1] * reach * height)
            obj.data = self.disc if spec["kind"] == "cylinder" else self.body_meshes[1]
            obj.material_slots[0].link = "OBJECT"
            obj.material_slots[0].material = self.distractor_materials[i]
            q = Quaternion(spec["rotation"])
            q.normalize()
            obj.rotation_quaternion = q
            obj.scale = spec["size"]
            nodes = self.distractor_materials[i].node_tree.nodes
            nodes["cells"].inputs["Scale"].default_value = spec["cells"]
            nodes["noise"].inputs["W"].default_value = spec["seed"]
            for k, element in enumerate(nodes["palette"].color_ramp.elements):
                element.color = srgb_to_linear(spec["colours"][k])
            threshold = 0.5 - spec["gap"] if spec["gap"] > 0 else 1.0
            for axis in ("X", "Y", "Z"):
                nodes[f"edge{axis}"].inputs[1].default_value = threshold
            _set(_bsdf(self.distractor_materials[i]), "Roughness", spec["roughness"])

    def _backdrop(self, sc: Scenario, cam_dir: Vector) -> None:
        bg = sc.background
        cam_distance = self.camera.location.length
        d = bg["distance"]
        self.backdrop.location = -cam_dir * d
        self.backdrop.rotation_quaternion = cam_dir.to_track_quat("Z", "Y")
        size = 2.4 * (d + cam_distance) * math.tan(self.camera.data.angle / 2) + 2
        self.backdrop.scale = (size, size, 1)
        self._backdrop_material(self.backdrop.data.materials[0], bg)
        if sc.table and sc.elevation > -5:
            self.table.hide_render = self.table.hide_viewport = False
            self.table.location = (0, 0, -sc.table["height"])
            self.table.rotation_quaternion = Quaternion()
            self.table.scale = (80, 80, 1)
            t = _bsdf(self.table.data.materials[0])
            _set(t, "Base Color", srgb_to_linear(sc.table["color"]))
            _set(t, "Roughness", 0.6)
        else:
            self.table.hide_render = self.table.hide_viewport = True

    def _backdrop_material(self, mat: bpy.types.Material, bg: dict) -> None:
        # The same backdrop as the last image (a video's frames): its nodes as they are
        if getattr(self, "_backdrop_of", None) == bg:
            return
        self._backdrop_of = dict(bg)
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        for node in [n for n in nodes if n.type not in ("BSDF_PRINCIPLED", "OUTPUT_MATERIAL")]:
            nodes.remove(node)
        bsdf = _bsdf(mat)
        _set(bsdf, "Roughness", 0.8)
        kind = bg["kind"]
        if kind == "image" and self.backgrounds:
            path = str(self.backgrounds[bg["image"] % len(self.backgrounds)])
            # One photo in memory at a time: thousands of them kept loaded would fill the RAM
            for old in [im for im in bpy.data.images if im.filepath and im.filepath != path and im.users == 0]:
                bpy.data.images.remove(old)
            image = bpy.data.images.load(path, check_existing=True)
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = image
            # The plane has no UVs: its generated coordinates span the photo over it
            coords = nodes.new("ShaderNodeTexCoord")
            links.new(coords.outputs["Generated"], tex.inputs["Vector"])
            links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
            self._glow(bsdf, tex.outputs["Color"], bg, links)
            return
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = bg["scale"] if kind == "procedural" else 2.0
        ramp = nodes.new("ShaderNodeValToRGB")
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        if kind == "procedural":
            colors = bg["colors"]
        elif kind == "skin":
            # A torso: skin with soft light and shade
            colors = [[c * 0.7 for c in bg["skin"]], bg["skin"], [min(1, c * 1.1) for c in bg["skin"]]]
        else:
            base = bg["colors"][0]
            grey = sum(base) / 3
            wall = [0.55 + 0.4 * (0.7 * grey + 0.3 * c) for c in base]
            colors = [[c * 0.93 for c in wall], wall, wall]
        ramp.color_ramp.elements[0].color = srgb_to_linear(colors[0])
        ramp.color_ramp.elements[1].color = srgb_to_linear(colors[1])
        mid = ramp.color_ramp.elements.new(0.5)
        mid.color = srgb_to_linear(colors[2])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        self._glow(bsdf, ramp.outputs["Color"], bg, links)

    @staticmethod
    def _glow(bsdf, colour, bg: dict, links) -> None:
        """A backdrop that shines (a window, a screen behind the cube: the cube against the light)."""
        glow = bg.get("glow", 0.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = glow
            if glow > 0 and "Emission Color" in bsdf.inputs:
                links.new(colour, bsdf.inputs["Emission Color"])

    def _lights(self, sc: Scenario, cam_dir: Vector) -> None:
        cam_az = math.atan2(cam_dir.x, -cam_dir.y)
        # The exposure scales every light (lamps and world, the only emitters) instead of being the view's:
        # the same pixels, as the view's exposure multiplies the scene-linear light by 2^EV, but a new
        # exposure with each switch of view transform leaked 1.5 MB per image (colour processors kept)
        gain = 2.0 ** sc.exposure
        for i, light in enumerate(self.lights):
            spec = sc.lights[i] if i < len(sc.lights) else None
            light.hide_render = light.hide_viewport = spec is None
            if spec is None:
                continue
            az = cam_az + math.radians(spec.azimuth)
            el = math.radians(spec.elevation)
            light.location = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * spec.distance
            light.rotation_mode = "QUATERNION"
            light.rotation_quaternion = (-light.location).to_track_quat("-Z", "Y")
            light.data.energy = spec.energy * gain
            light.data.size = spec.size
            light.data.color = [c * t for c, t in zip(kelvin_rgb(spec.kelvin), spec.tint, strict=True)]
        world = self.scene.world
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Color"].default_value = srgb_to_linear(sc.ambient)
            bg_node.inputs["Strength"].default_value = gain
        self.scene.view_settings.view_transform = sc.tone

    def _motion(self, sc: Scenario) -> None:
        """Pose at frame 1; with a shutter, keys at frames 0 and 2 give motion blur."""
        objs = [self.root, *self.bodies.values()]
        for obj in objs:
            # Clearing the animation data leaves its action behind (orphan, in memory): remove it,
            # or every image's keys pile up (2 MB per image, WSL froze after 30 000 renders)
            if obj.animation_data and obj.animation_data.action:
                bpy.data.actions.remove(obj.animation_data.action)
            obj.animation_data_clear()
        turning = self._turning_bodies(sc)
        cut = self._next_bodies(sc, turning)
        base_q = quat_to_bl(sc.orientation)
        angle = sc.turn.angle if sc.turn else 0.0
        angle_next = sc.turn_next.angle if sc.turn_next else 0.0
        self.scene.render.use_motion_blur = sc.shutter > 0
        self._slack(sc, turning, cut)
        if sc.shutter <= 0:
            self.root.rotation_quaternion = base_q
            if turning:
                self._set_turn(*turning, angle)
            if cut:
                self._set_turn(*cut, angle_next)
            return
        self.scene.render.motion_blur_shutter = sc.shutter
        spin_axis = Vector(sc.spin[:3]).normalized() if Vector(sc.spin[:3]).length > 1e-6 else Vector((0, 0, 1))
        for frame, sign in ((0, -1), (1, 0), (2, 1)):
            self.root.rotation_quaternion = Quaternion(spin_axis, math.radians(sign * sc.spin[3])) @ base_q
            self.root.keyframe_insert("rotation_quaternion", frame=frame)
            if turning:
                bodies, axis = turning
                direction = math.copysign(1.0, angle) if angle else 1.0
                self._set_turn(bodies, axis, angle + sign * direction * sc.turn_speed)
                for body in bodies:
                    body.keyframe_insert("location", frame=frame)
                    body.keyframe_insert("rotation_quaternion", frame=frame)
            if cut:
                bodies, axis = cut
                direction = math.copysign(1.0, angle_next) if angle_next else 1.0
                self._set_turn(bodies, axis, angle_next + sign * direction * sc.turn_speed)
                for body in bodies:
                    body.keyframe_insert("location", frame=frame)
                    body.keyframe_insert("rotation_quaternion", frame=frame)

    def _measure(self) -> tuple[list[bool], list[float], list[float], list[int]]:
        """Which tiles the camera sees, even almost edge-on: any of 13 points of a tile (its centre, 4 around it,
        8 near its edges), a corner of colour peeking past a finger is a sticker read; how squarely each face
        faces it; each tile's area in the image (share of the image's area); and how many of the 5 inner points
        it sees (the rule before, 3 needed: to check that a relabelled dataset draws the scenes it was drawn from)."""
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cam = self.camera.evaluated_get(depsgraph).matrix_world.translation
        visible, facing, area, inner = [], [0.0] * 6, [], []
        for i, tile in enumerate(self.tiles):
            mw = tile.evaluated_get(depsgraph).matrix_world
            thickness = self._thickness
            normal = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
            centre = mw @ Vector((0, 0, thickness / 2 + 1e-3))
            to_cam = (cam - centre).normalized()
            cos = normal.dot(to_cam)
            if i % 9 == 4:
                facing[i // 9] = cos
            hits, outer, projected = 0, 0, 0.0
            if cos > 0.02:
                corners = [world_to_camera_view(self.scene, self.camera, mw @ Vector((dx, dy, thickness / 2))) for dx, dy in
                           ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))]
                projected = abs(sum(a.x * b.y - b.x * a.y for a, b in zip(corners, corners[1:] + corners[:1], strict=True))) / 2
                for k, (dx, dy) in enumerate(TILE_POINTS):
                    p = mw @ Vector((dx, dy, thickness / 2 + 1e-3))
                    x, y, z = world_to_camera_view(self.scene, self.camera, p)
                    if z <= 0 or not (0.01 < x < 0.99 and 0.01 < y < 0.99):
                        continue
                    ray = p - cam
                    hit, location, _n, _i, obj, _m = self.scene.ray_cast(depsgraph, cam, ray.normalized(), distance=ray.length + 0.05)
                    if hit and (obj.name == tile.name or (location - p).length < 0.03):
                        if k < 5:
                            hits += 1
                        else:
                            outer += 1
            visible.append(hits + outer > 0)
            inner.append(hits)
            area.append(round(projected, 7))
        return visible, facing, area, inner
