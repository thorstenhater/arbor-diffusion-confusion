#!/usr/bin/env python3

import arbor as A
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from glob import iglob
import subprocess as sp

this = Path(__file__)
here = this.parent
mech = here / 'mechanisms'
mtime = this.stat().st_mtime

recompile = not (here / 'custom-catalogue.so').exists()
for fn in iglob(str(mech) + '/*.mod'):
	other = Path(fn).stat().st_mtime
	if other > mtime:
            recompile = True
if recompile:
    print("RECOMPILE")
    sp.run('arbor-build-catalogue custom mechanisms', shell=True)

class diffusionRecipe(A.recipe):
    def __init__(self, dx, r_soma, r_dend):
        A.recipe.__init__(self)
        self.the_props = A.neuron_cable_properties()
        self.the_props.catalogue.extend(A.load_catalogue("./custom-catalogue.so"), "")

        # diffusivity of particles
        self.diffusivity = 1
        self.the_props.set_ion("s", 1, 0, 0, self.diffusivity)
        self.the_props.set_ion("p", 1, 0, 0, self.diffusivity)

        self.ch_1 = 1000
        self.ch_2 = 500
        self.times_1 = [0.5, 1.5,]
        self.times_2 = [1.0]
        self.radius_soma = r_soma
        self.length_soma = 10
        self.radius_dendrite = r_dend
        self.length_dendrite = 20

        self.delta_x = dx

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):

        # cell morphology (consisting of cylindrical soma and dendrite)
        tree = A.segment_tree()
        tag_soma = 1
        tag_dendrite1 = 2
        tag_dendrite2 = 3

        sm = tree.append(A.mnpos,
                         A.mpoint(0, -self.length_soma/2, 0, self.radius_soma),
                         A.mpoint(0,  self.length_soma/2, 0, self.radius_soma),
                         tag=tag_soma)

        _ = tree.append(sm,
                        A.mpoint(0, self.length_soma/2, 0, self.radius_dendrite),
                        A.mpoint(0, self.length_dendrite+self.length_soma/2, 0, self.radius_dendrite),
                        tag=tag_dendrite1)

        _ = tree.append(sm,
                        A.mpoint(0, -self.length_soma/2, 0, self.radius_dendrite),
                        A.mpoint(0, -self.length_dendrite - self.length_soma/2, 0, self.radius_dendrite),
                        tag=tag_dendrite2)

        labels = A.label_dict({"soma_region": f"(tag {tag_soma})",
                               "dendrite1_region": f"(tag {tag_dendrite1})",
                               "dendrite2_region": f"(tag {tag_dendrite2})",
                               "soma": '(on-components 0.5 (region "soma_region"))',
                               "dendrite1_synapses": '(on-components 1.0 (region "dendrite1_region"))',
                               "dendrite2_synapses": '(on-components 1.0 (region "dendrite2_region"))'})

        self.area_soma_µm2     = 2 * np.pi * self.radius_soma     * self.length_soma
        self.area_dendrite_µm2 = 2 * np.pi * self.radius_dendrite * self.length_dendrite
        self.area_µm2          = self.area_soma_µm2 + 2 * self.area_dendrite_µm2

        self.volume_soma_µm3     = np.pi * self.radius_soma**2     * self.length_soma
        self.volume_dendrite_µm3 = np.pi * self.radius_dendrite**2 * self.length_dendrite
        self.volume_µm3          = self.volume_soma_µm3 + 2 * self.volume_dendrite_µm3

        print("       | Dendrite | Soma     | Total    |")
        print("-------+----------+----------+----------|")
        print(f"radius | {self.radius_dendrite:8.2f} | {self.radius_soma:8.2f} |")
        print(f"area   | {self.area_dendrite_µm2:8.2f} | {self.area_soma_µm2:8.2f} | {self.area_µm2:8.2f} |")
        print(f"volume | {self.volume_dendrite_µm3:8.2f} | {self.volume_soma_µm3:8.2f} | {self.volume_µm3:8.2f} |")
        print()
        print(f"Parameters diffusivity={self.diffusivity} dx={self.delta_x} rs={self.radius_soma} rd={self.radius_dendrite}")

        decor = (A.decor()
                  .discretization(A.cv_policy(f'(max-extent {self.delta_x})'))
                  # TODO: This shouldn't be needed, but is
                  .set_ion("s", int_con=0.0, diff=self.diffusivity)
                  .set_ion("p", int_con=0.0, diff=self.diffusivity)
                  .place('"dendrite1_synapses"', A.synapse("synapse_with_diffusion"), "syn_1")
                  .place('"dendrite2_synapses"', A.synapse("synapse_with_diffusion"), "syn_2")
                  .paint('(region "soma_region")', A.density("neuron_with_diffusion"))
                  .paint('(all)', A.density("neuron_with_diffusion_2")))

        # print(A.morphology(tree))

        return A.cable_cell(tree, decor, labels)

    def global_properties(self, kind):
        return self.the_props

    def event_generators(self, gid):
        ev_gens = [A.event_generator("syn_1",  self.ch_1, A.explicit_schedule(self.times_1)),
                   A.event_generator("syn_2", -self.ch_2, A.explicit_schedule(self.times_2))]
        return ev_gens

    def probes(self, gid):
        return [A.cable_probe_ion_diff_concentration('"soma"', "s"),
		A.cable_probe_ion_diff_concentration('"dendrite1_synapses"', "s"),
		A.cable_probe_ion_diff_concentration('"dendrite2_synapses"', "s"),

		# SV ASSIGNED
                A.cable_probe_density_state('"soma"', "neuron_with_diffusion", "sV"),
		A.cable_probe_point_state(0, "synapse_with_diffusion", "sV"),
                A.cable_probe_point_state(1, "synapse_with_diffusion", "sV"),

		# sV STATE
		A.cable_probe_density_state('"soma"', "neuron_with_diffusion", "sVs"),
		A.cable_probe_point_state(0, "synapse_with_diffusion", "sVs"),
                A.cable_probe_point_state(1, "synapse_with_diffusion", "sVs"),

		# ???
		A.cable_probe_density_state_cell("neuron_with_diffusion", "sV"),
		A.cable_probe_ion_diff_concentration_cell("s"),
		]

def run_sim(points, r_soma, r_dend, tx):
    dt_ = 0.01
    t_final = 5.00

    rec = diffusionRecipe(points, r_soma, r_dend)
    sim = A.simulation(rec)

    hdls = [sim.sample((0, i), A.regular_schedule(dt_)) for i,_ in enumerate(rec.probes(0))]

    sim.run(dt=dt_, tfinal=t_final)
    fg, axs = plt.subplots(2, 3, sharex=True, )

    scale = rec.delta_x / rec.length_dendrite
    volume = [rec.volume_soma_µm3, rec.volume_dendrite_µm3 * scale, rec.volume_dendrite_µm3 * scale]
    area = [rec.area_soma_µm2, rec.area_dendrite_µm2 * scale, rec.area_dendrite_µm2 * scale ]

    for ix in range(2):
        for iy in range(3): # columns of plot
            idx = iy + 3*ix
            hdl = hdls[idx]
            ax = axs[ix][iy]
            ax.set_xlim(0, t_final)
            for data, meta in sim.samples(hdl):
                ax.plot(data[:, 0], data[:, 1])

    axs[0][0].set_ylabel('Xd $(mol/l)$')
    axs[1][0].set_ylabel('Nd $(mol)$')
    # axs[2][0].set_ylabel('NdS $(mol)$')

    for ix, title in enumerate(["Soma", "Dendrite 1", "Dendrite 2"]):
	    # axs[0][ix].set_ylim(0, 0.1)
	    # axs[1][ix].set_ylim(0, 150)
	    axs[0][ix].set_title(title)
	    axs[-1][ix].set_xlabel('Time $(t/ms)$')

    fg.savefig(f'diff-dx={rec.delta_x}-rs={rec.radius_soma}-rd={rec.radius_dendrite}.png')

    for data, meta in sim.samples(hdls[9]):
        total = np.zeros_like(data[:, 0])
        for ix, loc in enumerate(meta):
            total += data[:, ix + 1]
        tx.plot(data[:, 0], total, label=f"Total particles dx={rec.delta_x} rs={rec.radius_soma} rd={rec.radius_dendrite}")
        print(f"Equilibrium")
        print(f" * Particles      | {total[-1]:10.4f}")
    for data, meta in sim.samples(hdls[10]):
        print(f" * Concentration  | {data[-1, 1]:10.4f}")
        print(f" * Particles'     | {data[-1, 1]*rec.volume_µm3:10.4f}")




fg, ax = plt.subplots()

for dx in [1, 2]:
    for rs in [10, 20]:
        for rd in [10, 20]:
            print(80*'=')
            run_sim(dx, rs, rd, ax)

ax.legend()
fg.savefig('totals.pdf')
