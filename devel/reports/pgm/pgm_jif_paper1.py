from matplotlib import rc
rc("font", family="serif", size=10)
rc("text", usetex=True)

import daft

figshape = (2.65, 3.8)
figorigin = (0.4, -0.4)

pgm = daft.PGM(figshape, origin=figorigin)

pgm.add_node(daft.Node("P(omega)", r"${\bf d}_{\rm train}$", 2.1, 3))
pgm.add_node(daft.Node("gal_props", r"$\omega_{n}$", 2.1, 2.2))
pgm.add_node(daft.Node("data", r"${\bf d}_{ni}$", 2.1, 1, observed=True))
pgm.add_node(daft.Node("PSF", r"$\Pi_{ni}$", 1, 1.5, observed=True))
pgm.add_node(daft.Node("noise", r"$\sigma^{\rm noise}_{ni}$", 1, 0.5, observed=True))

pgm.add_edge("P(omega)", "gal_props")
pgm.add_edge("gal_props", "data")
pgm.add_edge("PSF", "data")
pgm.add_edge("noise", "data")

pgm.add_plate(daft.Plate([0.5, 0.05, 2.2, 2.63], label=r"galaxies $n$", label_offset=[5, 137]))
pgm.add_plate(daft.Plate([0.6, 0.1, 2.0, 1.75], label=r"epochs $i$", label_offset=[70, 5]))

pgm.render()
pgm.figure.savefig("../jif_paper1/figure/pgm.png", dpi=220)



