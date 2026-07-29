import os
import random
import re
import shutil
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PSEUDO_DIR = "/content/pseudopot"

ATOMIC_MASS = {
    'Al': 26.98, 'Co': 58.93, 'Cr': 52.00, 'Cu': 63.55, 'Fe': 55.85,
    'Ni': 58.69, 'V': 50.94, 'Mo': 95.95, 'Ti': 47.87, 'W': 183.84,
    'Mn': 54.94, 'Nb': 92.91, 'Ta': 180.95, 'Zr': 91.22, 'Hf': 178.49,
    'Si': 28.085, 'C': 12.011, 'B': 10.81, 'N': 14.01, 'O': 16.00,
    'Y': 88.91, 'Sc': 44.96, 'Zn': 65.38, 'Mg': 24.31, 'P': 30.97,
    'Nd': 144.24, 'Ce': 140.12, 'Sm': 150.36, 'Gd': 157.25,
    'Sn': 118.71, 'Ag': 107.87, 'Au': 196.97, 'Ru': 101.07, 'Pd': 106.42,
    'Li': 6.94, 'Ca': 40.078, 'Ga': 69.723,
    'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Rh': 102.91
}

PSEUDOPOTENTIALS = {el: f"{el}.pbe-spn-kjpaw_psl.1.0.0.UPF" for el in ATOMIC_MASS.keys()}
PSEUDOPOTENTIALS.update({
    'Al': 'Al.pbe-n-kjpaw_psl.1.0.0.UPF', 'Cu': 'Cu.pbe-dn-kjpaw_psl.1.0.0.UPF',
    'Si': 'Si.pbe-n-kjpaw_psl.1.0.0.UPF', 'C': 'C.pbe-n-kjpaw_psl.1.0.0.UPF',
    'B': 'B.pbe-n-kjpaw_psl.1.0.0.UPF', 'N': 'N.pbe-n-kjpaw_psl.1.0.0.UPF',
    'O': 'O.pbe-n-kjpaw_psl.1.0.0.UPF', 'P': 'P.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Nd': 'Nd.pbe-spdfn-kjpaw_psl.1.0.0.UPF', 'Ce': 'Ce.pbe-spdfn-kjpaw_psl.1.0.0.UPF',
    'Sm': 'Sm.pbe-spdfn-kjpaw_psl.1.0.0.UPF', 'Gd': 'Gd.pbe-spdfn-kjpaw_psl.1.0.0.UPF',
    'Sn': 'Sn.pbe-dn-kjpaw_psl.1.0.0.UPF', 'Ag': 'Ag.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Au': 'Au.pbe-n-kjpaw_psl.1.0.0.UPF', 'Li': 'Li.pbe-s-kjpaw_psl.1.0.0.UPF',
    'Ca': 'Ca.pbe-spn-kjpaw_psl.1.0.0.UPF', 'Ga': 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF'
})

LATTICE_BCC = {'V': 3.03, 'Nb': 3.30, 'Ta': 3.30, 'Cr': 2.88, 'Mo': 3.15, 'W': 3.17, 'Fe': 2.87, 'Al': 3.25, 'Co': 2.83, 'Ni': 2.81, 'Cu': 2.88, 'Ti': 3.28, 'Mn': 2.88, 'Zr': 3.54, 'Hf': 3.50}
LATTICE_FCC = {'Al': 4.05, 'Ni': 3.52, 'Co': 3.54, 'Cu': 3.61, 'Fe': 3.59, 'V': 3.80, 'Nb': 4.15, 'Ta': 4.15, 'Cr': 3.60, 'Mo': 3.95, 'W': 4.00, 'Ti': 4.10, 'Mn': 3.65, 'Zr': 4.53, 'Hf': 4.50}

MAG_DEFAULTS = {'Fe': 0.50, 'Co': 0.35, 'Ni': 0.20, 'Cr': 0.10, 'Mn': 0.40, 'Nd': 0.30}

TARGET_ATOMS = 8
ECUTWFC = 40.0
ECUTRHO = 320.0
CONV_THR = "1.0d-5"
ELECTRON_MAXSTEP = 40
NSTEP = 30
DEGAUSS = 0.03

def parse_formula(formula_str):
    matches = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', str(formula_str).replace(' ', ''))
    comp = {el: float(amt) if amt else 1.0 for el, amt in matches}
    total = sum(comp.values())
    if total == 0:
        raise ValueError(f"Invalid formula provided: {formula_str}")
    return {k: v / total for k, v in comp.items()}

def get_target_atom_counts(composition_dict):
    exact = {el: frac * TARGET_ATOMS for el, frac in composition_dict.items()}
    int_counts = {el: int(val) for el, val in exact.items()}
    remainders = {el: exact[el] - int_counts[el] for el in exact}
    missing = TARGET_ATOMS - sum(int_counts.values())

    sorted_rem = sorted(remainders.items(), key=lambda x: (x[1], x[0]), reverse=True)
    for i in range(missing):
        int_counts[sorted_rem[i][0]] += 1

    final_counts = {el: c for el, c in int_counts.items() if c > 0}
    assert sum(final_counts.values()) == TARGET_ATOMS
    return dict(sorted(final_counts.items(), key=lambda item: -item[1]))

def get_lattice_params(comp_dict):
    a_bcc = sum(f * LATTICE_BCC.get(el, 3.0) for el, f in comp_dict.items())
    a_fcc = sum(f * LATTICE_FCC.get(el, 3.6) for el, f in comp_dict.items())
    return a_bcc, a_fcc

def generate_coords(phase):
    coords = []
    if phase == 'FCC':
        base = [('corner', (0,0,0)), ('face', (0.5,0.5,0)), ('face', (0.5,0,0.5)), ('face', (0,0.5,0.5))]
        for k in range(2):
            for tag, (bx, by, bz) in base:
                coords.append((tag, (bx, by, (k+bz)/2.0)))
    elif phase in ['BCC', 'B2']:
        base = [('corner', (0,0,0)), ('center', (0.5,0.5,0.5))]
        for i in range(2):
            for j in range(2):
                for tag, (bx, by, bz) in base:
                    coords.append((tag, ((i+bx)/2.0, (j+by)/2.0, bz)))
    return coords

def assign_atoms(coords, atom_counts, phase):
    atom_list = [el for el, count in atom_counts.items() for _ in range(count)]
    if phase in ['FCC', 'BCC']:
        random.shuffle(atom_list)
        return list(zip(atom_list, [c[1] for c in coords]))
    elif phase == 'B2':
        centers = [c[1] for c in coords if c[0] == 'center']
        corners = [c[1] for c in coords if c[0] == 'corner']
        ordered_coords = centers + corners
        return list(zip(atom_list, ordered_coords))

def write_espresso_in(filepath, structure, phase, alloy_id, a, atom_counts):
    unique_elements = list(set([el for el, pos in structure]))
    ntyp = len(unique_elements)
    kpoints = "4 4 4 0 0 0"

    if phase == 'FCC':
        a_cell, b_cell, c_cell = 1.0*a, 1.0*a, 2.0*a
    elif phase in ['BCC', 'B2']:
        a_cell, b_cell, c_cell = 2.0*a, 2.0*a, 1.0*a

    cell_matrix = f"  {a_cell:.6f}  0.0  0.0\n  0.0  {b_cell:.6f}  0.0\n  0.0  0.0  {c_cell:.6f}"
    prefix_name = f"alloy{alloy_id}_{phase.lower()}"

    with open(filepath, 'w') as f:
        f.write(f"&CONTROL\n  calculation = 'vc-relax',\n  nstep = {NSTEP},\n  tstress = .true.,\n  tprnfor = .true.,\n")
        f.write("  disk_io = 'low',\n  wf_collect = .false.,\n")
        f.write(f"  pseudo_dir = '{PSEUDO_DIR}',\n  outdir = './out/',\n  prefix = '{prefix_name}',\n/\n")

        f.write(f"&SYSTEM\n  ibrav = 0,\n  nat = {TARGET_ATOMS},\n  ntyp = {ntyp},\n")
        f.write(f"  ecutwfc = {ECUTWFC},\n  ecutrho = {ECUTRHO},\n")
        f.write(f"  occupations = 'smearing',\n  smearing = 'gaussian',\n  degauss = {DEGAUSS},\n  nspin = 2,\n")

        for idx, el in enumerate(unique_elements, start=1):
            if el in MAG_DEFAULTS:
                f.write(f"  starting_magnetization({idx}) = {MAG_DEFAULTS[el]},\n")

        f.write(f"/\n&ELECTRONS\n  mixing_beta = 0.3,\n  conv_thr = {CONV_THR},\n")
        f.write(f"  electron_maxstep = {ELECTRON_MAXSTEP},\n  diagonalization = 'david',\n/\n")

        f.write("&IONS\n  ion_dynamics = 'bfgs',\n/\n")
        f.write("&CELL\n  cell_dynamics = 'bfgs',\n/\n")

        f.write("ATOMIC_SPECIES\n")
        for el in unique_elements:
            f.write(f"  {el:2s} {ATOMIC_MASS[el]:6.2f}  {PSEUDOPOTENTIALS[el]}\n")

        f.write(f"\nCELL_PARAMETERS {{angstrom}}\n{cell_matrix}\n")
        f.write("\nATOMIC_POSITIONS {crystal}\n")
        for el, (x, y, z) in structure:
            f.write(f"  {el:2s} {x:.6f} {y:.6f} {z:.6f}\n")
        f.write(f"\nK_POINTS {{automatic}}\n  {kpoints}\n")

def main():
    ALLOY_COMPOSITIONS = {
        "GEN_0012457": "Mo12.5Ti16.8Nb18.4Zr18.3Ta18.7Hf15.4",
        "GEN_0023586": "Co18.1Fe20.2Ni18.0Cr19.8Mn23.9",
        "GEN_0050205": "Al11.6Fe17.5Ni23.9Cr17.1Mo10.0Ti19.9",
        "GEN_0064391": "Al21.4Co8.3Fe18.6Ni14.0Cr20.5Mo17.3",
        "GEN_0103655": "Co15.2Fe14.1Ni15.2Mo30.6V24.9"
    }

    output_dir = "Alloy_Inputs"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []

    for alloy_id, formula in ALLOY_COMPOSITIONS.items():
        safe_formula = formula.replace(' ', '')
        comp_dict = parse_formula(safe_formula)
        atoms_tgt = get_target_atom_counts(comp_dict)
        a_bcc, a_fcc = get_lattice_params(comp_dict)

        alloy_dir = os.path.join(output_dir, f"{alloy_id}_{safe_formula}")
        os.makedirs(alloy_dir, exist_ok=True)

        with open(os.path.join(alloy_dir, "metadata.txt"), "w") as mf:
            mf.write(f"Alloy ID: {alloy_id}\nFormula: {safe_formula}\n")
            mf.write(f"8-Atom Supercell Content: {atoms_tgt}\n")

        phases = ['FCC', 'BCC', 'B2']

        for phase in phases:
            coords = generate_coords(phase)
            struct = assign_atoms(coords, atoms_tgt, phase)
            file_name = f"{alloy_id}_{phase.lower()}.in"
            in_file = os.path.join(alloy_dir, file_name)

            a_val = a_fcc if phase == 'FCC' else a_bcc
            write_espresso_in(in_file, struct, phase, alloy_id, a_val, atoms_tgt)

        summary_data.append({
            'ID': alloy_id,
            'Formula': safe_formula,
            'Elements': "-".join([f"{k}{v}" for k, v in atoms_tgt.items()]),
            'Status': '3-Phase Generated'
        })

    pd.DataFrame(summary_data).to_csv("alloy_summary.csv", index=False)

    zip_filename = "Alloy_Inputs_Archive"
    shutil.make_archive(zip_filename, 'zip', output_dir)

    try:
        from google.colab import files
        files.download(f"{zip_filename}.zip")
        files.download("alloy_summary.csv")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
