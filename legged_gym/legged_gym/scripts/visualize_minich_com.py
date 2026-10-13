"""Plot URDF link centers of mass without ROS or Isaac Gym.

Run in gym4; add --show for a rotatable Matplotlib 3D window.
The skeleton shows joint origins, not the visual mesh or a simulated pose.
Magenta crosses are symmetry references, not corrected physical parameters.
"""

import argparse
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def vector(element, attribute, default="0 0 0"):
    return np.fromstring(element.get(attribute, default) if element is not None else default, sep=" ")


def rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def origin_transform(element):
    result = np.eye(4)
    r, p, y = vector(element, "rpy")
    result[:3, :3] = (rotation(np.array([0., 0., 1.]), y)
                      @ rotation(np.array([0., 1., 0.]), p)
                      @ rotation(np.array([1., 0., 0.]), r))
    result[:3, 3] = vector(element, "xyz")
    return result


def default_angles():
    # Read the literal configuration without importing the simulator packages.
    tree = ast.parse((ROOT / "legged_gym/envs/minich/minich_config.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "default_joint_angles" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("default_joint_angles literal not found")


def load_model(path, angles):
    robot = ET.parse(path).getroot()
    joints = list(robot.findall("joint"))
    children = {j.find("child").get("link") for j in joints}
    roots = {link.get("name") for link in robot.findall("link")} - children
    if len(roots) != 1:
        raise ValueError("Expected a single URDF root")
    transforms = {roots.pop(): np.eye(4)}
    while joints:
        remaining = []
        for joint in joints:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                remaining.append(joint)
                continue
            transform = origin_transform(joint.find("origin"))
            motion = np.eye(4)
            kind = joint.get("type")
            q = angles.get(joint.get("name"), 0.)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = rotation(vector(joint.find("axis"), "xyz", "1 0 0"), q)
            elif kind == "prismatic":
                motion[:3, 3] = vector(joint.find("axis"), "xyz", "1 0 0") * q
            elif kind != "fixed":
                raise ValueError(f"Unsupported joint type: {kind}")
            transforms[joint.find("child").get("link")] = transforms[parent] @ transform @ motion
        if len(remaining) == len(joints):
            raise ValueError("Unresolved joint tree")
        joints = remaining
    coms = {}
    for link in robot.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            local = vector(inertial.find("origin"), "xyz")
            name = link.get("name")
            world = transforms[name] @ np.r_[local, 1.]
            coms[name] = {"local_m": local.tolist(), "body_m": world[:3].tolist(),
                          "mass_kg": float(inertial.find("mass").get("value"))}
    return transforms, coms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ROOT / "resources/robots/mini_cheetah/urdf/mini_cheetah.urdf")
    parser.add_argument("--pose", choices=("default", "zero"), default="default")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/minich_com_visualization")
    parser.add_argument("--show", action="store_true", help="Open the interactive figure; requires a desktop display")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    transforms, coms = load_model(args.urdf, default_angles() if args.pose == "default" else {})
    fig = plt.figure(figsize=(17, 7))
    axes = [fig.add_subplot(131, projection="3d"), fig.add_subplot(132), fig.add_subplot(133)]
    colors = dict(FL="#0072B2", FR="#D55E00", RL="#009E73", RR="#8C564B")
    reference_color = "#CC0099"
    discrepancies = {}

    def project(points, ax, **kwargs):
        points = np.atleast_2d(points)
        if ax is axes[0]:
            return ax.plot(*points.T, **kwargs)
        indexes = (1, 2) if ax is axes[1] else (0, 1)
        return ax.plot(points[:, indexes[0]], points[:, indexes[1]], **kwargs)

    for leg, color in colors.items():
        chain = np.array([transforms[f"{leg}_{part}"][:3, 3] for part in ("hip", "thigh", "calf", "foot")])
        for ax in axes:
            project(chain, ax, color=color, marker=".", lw=2, alpha=.65)
            for part, marker in (("hip", "o"), ("thigh", "s")):
                actual = np.array(coms[f"{leg}_{part}"]["body_m"])
                project(actual, ax, color=color, marker=marker, ms=9, linestyle="None")
                if leg.endswith("R"):
                    reference = np.array(coms[f"{leg[0]}L_{part}"]["body_m"]) * [1, -1, 1]
                    discrepancies[f"{leg}_{part}"] = float(np.linalg.norm(actual - reference))
                    project(reference, ax, color=reference_color, marker="x", ms=10, mew=2, linestyle="None")
                    project([actual, reference], ax, color=reference_color, linestyle="--", lw=1.5)
        axes[2].annotate(leg, chain[-1, :2], xytext=(4, 4), textcoords="offset points", color=color, weight="bold")

    # Connect the four hip joint origins to outline the body attachment frame.
    body = np.array([transforms[f"{leg}_hip"][:3, 3] for leg in ("FL", "FR", "RR", "RL", "FL")])
    for ax in axes:
        project(body, ax, color="gray", lw=1.5, linestyle=":")
        ax.grid(alpha=.25)
    axes[0].set(xlabel="X: forward (m)", ylabel="Y: left (m)", zlabel="Z: up (m)", title="3D joint skeleton + link COM")
    axes[0].view_init(elev=23, azim=-50)
    axes[0].set_box_aspect((.6, .5, .5))
    axes[1].set(xlabel="Y: left-positive (m)", ylabel="Z: up (m)", title="Front projection (front/rear legs overlap)")
    axes[2].set(xlabel="X: forward (m)", ylabel="Y: left (m)", title="Top projection")
    for ax in axes[1:]:
        ax.set_aspect("equal", adjustable="box")
        ax.axvline(0, color="gray", lw=.6)
        ax.axhline(0, color="gray", lw=.6)
    handles = [Line2D([], [], color=c, lw=2, label=leg) for leg, c in colors.items()]
    handles += [Line2D([], [], color="black", marker=m, linestyle="None", label=label)
                for m, label in (("o", "Hip COM"), ("s", "Thigh COM"))]
    handles += [Line2D([], [], color=reference_color, marker="x", linestyle="--", label="Right-side mirror of left COM")]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(.5, .08))
    fig.suptitle(f"Mini Cheetah URDF centers of mass | {args.pose} joint pose | body frame", fontsize=16)
    local_y = {part: [1000 * coms[f"{leg}_{part}"]["local_m"][1] for leg in colors]
               for part in ("hip", "thigh")}
    hip_gap = max(v for k, v in discrepancies.items() if k.endswith("hip")) * 1000
    thigh_gap = max(v for k, v in discrepancies.items() if k.endswith("thigh")) * 1000
    fig.text(.5, .035, f"Local Y [FL, FR, RL, RR] mm: hip {local_y['hip']}; thigh {local_y['thigh']}. "
             f"Max mirror gaps: {hip_gap:.1f} / {thigh_gap:.1f} mm.\n"
             "Magenta = symmetry reference only (not CAD truth). Skeleton is not the mesh; pose is not simulated equilibrium.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .22, 1, .94), w_pad=3)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"com_{args.pose}.png"
    fig.savefig(destination, dpi=170)
    (args.output / f"com_{args.pose}.json").write_text(json.dumps(
        {"urdf": str(args.urdf.resolve()), "pose": args.pose, "coms": coms,
         "right_vs_mirrored_left_distance_m": discrepancies}, indent=2) + "\n")
    print(f"Saved {destination}")
    print("Right COM vs mirrored left COM distance (m):", discrepancies)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
