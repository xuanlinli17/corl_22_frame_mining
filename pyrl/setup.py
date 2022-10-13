from setuptools import setup, find_packages
from pkg_resources import DistributionNotFound, get_distribution
import os, re

def choose_requirement(primary, secondary):
    """If some version of primary requirement installed, return primary, else return secondary."""
    try:
        name = re.split(r"[!<>=]", primary)[0]
        get_distribution(name)
    except DistributionNotFound:
        return secondary
    return str(primary)


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys, re, os.path as osp

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                if not (os.getenv("WITH_CUDA", "1") == "0" and "torchsparse" in line):
                    info["package"] = line
            else:
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath, dep_link=False):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    if "--find-links" in line:
                        if dep_link:
                            yield line.replace("--find-links", "").strip()
                    elif not dep_link:
                        for info in parse_line(line):
                            yield info

    def gen_packages_items(dep_link=False):
        if not osp.exists(require_fpath):
            return
        for info in parse_require_file(require_fpath, dep_link):
            if dep_link:
                yield info
            else:
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    dep_links = list(gen_packages_items(True))
    return packages, dep_links


install_requires, dependency_links = parse_requirements()

try:
    import cv2  # NOQA: F401

    major, minor, *rest = cv2.__version__.split(".")
    if int(major) < 3:
        raise RuntimeError(f"OpenCV >=3 is required but {cv2.__version__} is installed")
except ImportError:
    CHOOSE_INSTALL_REQUIRES = [("opencv-python-headless>=3", "opencv-python>=3")]
    for main, secondary in CHOOSE_INSTALL_REQUIRES:
        install_requires.append(choose_requirement(main, secondary))



setup(
    name="pyrl",
    version="1.0.0",
    install_requires=install_requires,
    dependency_links=dependency_links,
    packages=find_packages(),
    include_package_data=True,
    author="Anonymous",
    zip_safe=False,
)
