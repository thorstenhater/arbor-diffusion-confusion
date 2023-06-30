#!/usr/bin/env python3

import arbor as A
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
        self.diffusivity = 1#e-16
        self.the_props.set_ion("s", 1, 0, 0, self.diffusivity)
        self.the_props.set_ion("p", 1, 0, 0, self.diffusivity)

        self.ch_0 = 420
        self.ch_1 = 1000
        self.ch_2 = 500
        self.times_0 = [0.75]
        self.times_1 = [0.5, 1.5,]
        self.times_2 = [1.0]
        self.radius_soma = r_soma
        self.length_soma = 10
        self.radius_dendrite = r_dend
        self.length_dendrite = 20

        self.delta_x = dx

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, _):
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
                        A.mpoint(0, self.length_soma/2, 0, self.radius_dendrite),
                        A.mpoint(0, self.length_dendrite + self.length_soma/2, 0, self.radius_dendrite),
                        tag=tag_dendrite2)

        labels = A.label_dict({"soma_region": f"(tag {tag_soma})",
                               "dendrite1_region": f"(tag {tag_dendrite1})",
                               "dendrite2_region": f"(tag {tag_dendrite2})",
                               "soma": '(on-components 0.5 (region "soma_region"))',
                               "dendrite1_synapses": '(on-components 0.5 (region "dendrite1_region"))',
                               "dendrite2_synapses": '(on-components 0.5 (region "dendrite2_region"))'})

        self.area_soma     = 2 * np.pi * self.radius_soma     * self.length_soma
        self.area_dendrite = 2 * np.pi * self.radius_dendrite * self.length_dendrite
        self.area          = self.area_soma + 2 * self.area_dendrite

        self.volume_soma     = np.pi * self.radius_soma**2     * self.length_soma
        self.volume_dendrite = np.pi * self.radius_dendrite**2 * self.length_dendrite
        self.volume          = self.volume_soma + 2 * self.volume_dendrite

        print("       | Dendrite | Soma     | Total    |")
        print("-------+----------+----------+----------|")
        print(f"radius | {self.radius_dendrite:8.2f} | {self.radius_soma:8.2f} |")
        print(f"area   | {self.area_dendrite:8.2f} | {self.area_soma:8.2f} | {self.area:8.2f} |")
        print(f"volume | {self.volume_dendrite:8.2f} | {self.volume_soma:8.2f} | {self.volume:8.2f} |")
        print()
        print(f"Parameters diffusivity={self.diffusivity} dx={self.delta_x} rs={self.radius_soma} rd={self.radius_dendrite}")

        decor = (A.decor()
                 .discretization(A.cv_policy(f'(max-extent {self.delta_x})'))
                 # TODO: This shouldn't be needed, but is
                 .set_ion("s", int_con=0.0, diff=self.diffusivity)
                 .set_ion("p", int_con=0.0, diff=self.diffusivity)
                 .place('"dendrite1_synapses"', A.synapse("synapse_with_diffusion"), "syn_1")
                 .place('"dendrite2_synapses"', A.synapse("synapse_with_diffusion"), "syn_2")
                 .place('"soma"', A.synapse("synapse_with_diffusion"), "syn_0")
                 .paint('(all)', A.density("neuron_with_diffusion")))

        # print(A.morphology(tree))

        return A.cable_cell(tree, decor, labels)

    def global_properties(self, _):
        return self.the_props

    def event_generators(self, _):
        return [A.event_generator("syn_0",  self.ch_0, A.explicit_schedule(self.times_0)),
            A.event_generator("syn_1",  self.ch_1, A.explicit_schedule(self.times_1)),
        A.event_generator("syn_2", -self.ch_2, A.explicit_schedule(self.times_2))]

    def probes(self, _):
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
                A.cable_probe_point_state(2, "synapse_with_diffusion", "sV"),

                # Probe DENSITY state at different places
                A.cable_probe_density_state('"soma"', "neuron_with_diffusion", "sV"),
                A.cable_probe_density_state('"dendrite1_synapses"', "neuron_with_diffusion", "sV"),
                A.cable_probe_density_state('"dendrite2_synapses"', "neuron_with_diffusion", "sV"),
        ]

def run_sim(points, r_soma, r_dend, tx):
    dt_ = 0.01
    t_final = 5.00

    rec = diffusionRecipe(points, r_soma, r_dend)
    sim = A.simulation(rec)

    hdls = [sim.sample((0, i), A.regular_schedule(dt_)) for i,_ in enumerate(rec.probes(0))]

    sim.run(dt=dt_, tfinal=t_final)
    fg, axs = plt.subplots(2, 3, sharex=True, )
    for ix in range(2):
        for iy in range(3): # columns of plot
            idx = iy + 3*ix
            hdl = hdls[idx]
            ax = axs[ix][iy]
            ax.set_xlim(0, t_final)
            for data, meta in sim.samples(hdl):
                ax.plot(data[:, 0], data[:, 1])

    # now add point state/density state to relevant panels
    ax = axs[1][0]
    for data, meta in sim.samples(hdls[11]):
            ax.plot(data[:, 0], data[:, 1], ls=':')

    ax = axs[1][1]
    for data, meta in sim.samples(hdls[13]):
            ax.plot(data[:, 0], data[:, 1], ls=':')

    ax = axs[1][2]
    for data, meta in sim.samples(hdls[12]):
            ax.plot(data[:, 0], data[:, 1], ls=':')

    for data, meta in sim.samples(hdls[7]):
            print(meta)
    for data, meta in sim.samples(hdls[8]):
            print(meta)

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
        for ix, _ in enumerate(meta):
            total += data[:, ix + 1]
            tx.plot(data[:, 0], total, label=f"Total particles dx={rec.delta_x} rs={rec.radius_soma} rd={rec.radius_dendrite}")
        print(f"Equilibrium")
        print(f" * Particles      | {np.max(total):10.4f}")
    for data, meta in sim.samples(hdls[10]):
            print(f" * Concentration  | {data[-1, 1]:10.4f}")
            print(f" * Particles'     | {data[-1, 1]*rec.volume:10.4f}")


fg, ax = plt.subplots()
for dx in [1, 2]:
    for rs in [10, 20]:
        for rd in [10, 20]:
            print(80*'=')
            run_sim(dx, rs, rd, ax)

ax.legend()
fg.savefig('totals.pdf')
