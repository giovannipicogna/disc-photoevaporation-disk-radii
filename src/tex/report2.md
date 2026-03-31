---
marp: true
---

# Reviewer's Comments:

**I appreciate the authors’ substantial efforts in revising the manuscript and addressing the comments in the previous report. The revised version now states the main contribution of the paper more clearly: namely, the development of a prescription that cuts off the mass-loss rate at the surface-density cut-off radius, with compact disks as the primary application. I agree that this is a useful point to communicate to the community.**

**That said, I still have concerns about several newly added parts of the revised manuscript. Some statements may be misleading, and others are not fully convincing from a physical standpoint. In addition, a few of my previous comments have not been addressed sufficiently. I also find that some of the new material does not provide sufficient information to understand the adopted methodology and results unambiguously.**

**For these reasons, I regret that I still cannot recommend the manuscript for publication in its current form, although I do find that the manuscript has significantly improved. I strongly encourage the authors to address the points raised above, as well as the detailed comments below.**

**Below, I first respond to the authors’ replies to my previous report, and then provide new comments on the updated portions of the manuscript.**

**1. Overestimated mass-loss rate (previous comment #1):**

**I agree that reducing the mass-loss rates by a factor of 10 from the hydrodynamic simulations is a reasonable and pragmatic choice for the population synthesis. However, I am not fully convinced by the explanation that attributes the high mass-loss rates primarily to general “uncertainties” (including the discussion following Eq. 10).**

**The ∼10x higher mass-loss rates appear more directly related to the neglect of O–H collisional excitation, for which I see no physical justification, rather than to uncertainties in parameters such as chemistry/metallicity or the input spectra. While mass-loss rates can vary with model parameters, this level of enhancement seems different from the factor-of-a-few variations typically seen among other radiation-hydrodynamics models that include this physics. In that sense, the authors’ rates appear to fall outside the usual range of model-dependent uncertainty, even qualitatively.**

**If the authors wish to argue that their results are within such model variations, it would be helpful to demonstrate that similarly high mass-loss rates (∼1e-8 Msun/yr) persist when O–H excitation is included. Otherwise, the current discussion risks conflating parameter dependence with the consequences of a specific (unphysical) modeling assumption, which may be confusing to readers.**

**To clarify this point, I suggest one of the following approaches:**

**(a) remove the discussion of parameter-dependent variations and simply state, for example, “The model does not include O–H excitation, which leads to ∼10x higher mass-loss rates (Sellek et al. 2024); we therefore rescale the rates by a global factor,” or**

**(b) retain the discussion but recompute the mass-loss rates including O–H excitation to show that values of order 1e-8 Msun/yr can still be obtained.**

**I emphasize that the origin of the authors’ high mass-loss rates appears fundamentally different from the factor-of-a-few variations seen among other models, and distinguishing these two effects more clearly is important for the community.**

We thank the referee for pointing this out. Sellek et al. (2024) indeed showed that the lack of O-H collisions were the main drivers of the discrepancy for the same parameters choice, and we highlight this point better in the discussion following eq. 10.
However, we don't feel that removing the discussion is correct either. As shown in Wölfer et al. (2019, see their Fig. 6), a decreased disc metallicity can boost by almost an order of magnitude the mass-loss rates.
We want thus to stretch that keeping a constant reduction of a factor 10 with respect to Picogna et al. (2019, 2021) is not the best approach either, since this will change with time (as the dust is settling reducing the metallicity in the disc atmosphere), but for now it is indeed the best option to reconcile the wind disc models with observed disc lifetimes.
To reflect this view, we show now the main results (Figs. 4, 6, and the new Figure 7) for the reduced photoevaporation rates, while keeping the originale values for comparison.

**2. Prescription of mass-loss rates (previous comment #2)**

**I thank the authors for clarifying the meanings of the expressions; they are now much clearer. While the final choice of notation is up to the authors, I have one further suggestion to avoid potential confusion. It may be helpful to use symbols other than dot{M} on the left-hand sides of Eqs. 6 and 8. The symbol dot{M} typically denotes a quantity with physical units of mass per time, whereas the normalization factors in these equations appear to be dimensionless (unless I am mistaken). As written, some readers may wonder whether the quantities are truly dimensionless or whether physical units have been inadvertently omitted from the right-hand sides.**

We changed the equations 6 and 8 accordingly.

**3. Eq.1: The second equality is incorrect, as the latter expression includes an extra factor of (gamma-1)/(2gamma). In addition, the simultaneous use of gamma = 5/3 and mu = 2.35 is confusing: the former corresponds to atomic gas, while the latter is appropriate for molecular gas. Relatedly, the symbol gamma is also used in Eq. 11 with a different meaning, so the notation should be clarified to avoid confusion.**

We thank the referee to point out this inconsistency. We corrected the equation and added a small paragraph to explain the dependance of the gravitational radius on the gas properties.

**4. Eqs.9 and 10: Please include physical units inside the logarithms, e.g., log L_X → log (L_X / erg s^{-1}). In addition, please specify the physical units on the right-hand side of Eq. 9.**

We changed the equations 9 and 10 accordingly.

**5. Section 3.1, first paragraph (cut-off): I am slightly confused by the sentence beginning “The cut-off …”. Does the simulation include viscosity, or is this statement intended to justify neglecting viscosity in the hydrodynamic simulations? If viscosity is included, it would be helpful to state this explicitly in Section 2.2.**

Yes, the hydrodynamical simulations adopt an $\alpha$ viscosity of $0.001$. The cut-off radius refers to the initial disc mass, that is exponentially cutted at that specific location. During few hundred orbits (at 10 au) the cut-off radius does not evolve signifincantly. We added the adopted viscosity value in Sec 2.2 as suggested.

**6. Section 3.1, first paragraph (streamlines): If the streamlines curve back toward the midplane, this implies that gas is transported to larger radii, resulting in mass gain there. What are the implications of this effect for long-term disk evolution? I also note that this behavior may be related to the neglect of scattered or diffuse radiation in the model (also see the next comment).**

Indeed there is some material that is recirculated outside the cut-off radius, as shown in Figure 2 with the unbound region highlighted with a dashed red line. However, this effect is marginal at best. The amount of material recirculated can be estimated from the tail of the cumulative mass loss rate. Its effect might be more pronounced for cut-off radii smaller than 10 au where the cumulative mass-loss rate scales as r^3, but even in this case it would be less than 1e-9, enhanching only slightly the viscous expansion of the disc. If one would consider environment effects, like external photoevaporation, this loosely bound material would be nevertheless removed from the system. We added a paragraph in Section 3.1 to highlight this point.

**7. Eq.14: I find this equation confusing. Substituting $r_1$ into Eq. 6 removes its radial dependence. Since the radial dependence of Eq. 4 is derived from Eq. 6, it is unclear how Eq. 4 is modified as a result. From Figure 4, I infer that $\dot{\Sigma}_w$ is simply cut off at $r_1$. If so, I do not agree with this treatment, because in reality photoevaporation can occur beyond $r_1$ due to scattered or diffuse radiation reaching the shadowed region, like in the view of Hollenbach+94. This assumption would therefore lead to overestimating the disk lifetimes and therefore the main conclusions.**

This is exactly the point, and one of the major takeaway from the paper. In the Hollenbach view the diffuse EUV field was originating from close dense bound material. It is true that our heating prescription based on the ionization parameter assumes that the main source of heating is direct irradiation. However, the diffuse material close to the disc outer edge is much less dense than in the Hollenbach picture, and the resulting diffuse EUV field is negligible compared to outer environmental effects. When discussing the long-term disc evolution in the population synthesis we take into account also the effect of external photoevaporation, showing that (for the mild environment of the nearby SFR) it has a marginal effect on the disc lifetime. We added a paragraph at the end of Section 3.1 to further discuss this point.

**8. Section 3.2, first paragraph: I am confused by the newly added statement: "we fixed the cut-off radius seen by the internal photoevaporation for the compact discs, while allowing it to radially spread for the extended ones." My initial interpretation was that $\dot{\Sigma}_w$ is assumed to be time-independent, but does this instead mean that the authors artificially suppress viscous spreading in the compact-disk models? If so, this requires further explanation. However, Figure 4 appears to show radial spreading even for compact disks, suggesting that they are in fact allowed to spread viscously. This seems inconsistent with the introduction, where compact disks are defined as not radially spreading. If the authors are using viscous disk evolution as a proxy for MHD-wind-driven evolution, I don’t think it’s a valid approximation, since MHD winds can efficiently transport mass inward and refill gaps. In any case, a clear justification is needed for adopting this approach. To avoid these confusions, further clarification is needed regarding the definitions of "compact" and "extended" disks, both within this paper and in the general context. Moreover, the term “extended disks” itself may be confusing, since such disks can become compact through outside-in dispersal (as shown in Figure 4). Updating the terminology may help.**

We thank the referee for this observation and realized the confusion introduced by stating that we suppressed viscous spreading.
We shifted to a more physically based set of simulations, in which we included or not the effect of external photoevaporation to understand whether it was enough to reduce viscous spreading. Furthermore we compared the case where the internal photoevaporation is limited to the cut-off radius or is not. The new Fig. 4 shows that for compact discs (i.e. discs with an initial cut-off radius less than 20-30 au) the viscous spreading is hindered by the internal photoevaporation (even after reducing it by a factor 10) when it is not limited to the cut-off radius, which leads to an outside-in disc dispersal. When the proposed prescription is adopted, the disc viscously spread and is eventually dispersed from the inside-out. For large discs (i.e. with initial cut-off radius greater than 30 au) the internal photoevaporation cannot prevent disc spreading even in the unconstrained scenario. The disc in this case is dispersed usually from a mixed insed-out and outside-in fashion. We clarified throughout the text the nomenclature of compact and large (or extended) discs.

**9. Related to the above two comments (and previous comment #3), I also do not see a clear justification for assuming that the radius-scaled $\dot{\Sigma}_w$ remains constant in time. In viscous disks—and even in MHD-wind–driven disks—the location of the shadowed region can evolve radially as the disk evolves. This time dependence could significantly affect the overall disk lifetime and should be addressed or justified.**

As previously stated, the shadowed region, especially for radially extended discs, has a very low density and the EUV diffuse field generated is order of magnitude lower than the direct X-ray irradiation. The effect of external photoevporation introduced in the population synthesis should in most of the cases dominate over the internal diffuse EUV field. The mass-loss rate is nevertheless increasing in time as the disc viscously spread since the cut-off radius is updated at every time-step.

**10. Figure 4: (1) What does the number in parentheses in each panel title represent? Is it the disk lifetime? If so, why are all values very short (< 1 Myr)? (2) How is the disk lifetime defined—by the cessation of accretion, or by an infrared diagnostic? (3) Is the term “critical radius” in the legend intended to mean the “cut-off radius”? (4) Given the stated motivation of the paper, why is there no example of a compact disk that survives for a relatively long time?**

The values in the old Figure 4 were referring to a very specific parameter space not representative of the disc population with large viscosity and large internal photoevaporation rates. We updated the Figure including more representative values. The disc lifetime is defined as the time at which the maximum surface density falls below 1e-2 g/cm^2 to mock an observational limit. We realized that this information is very important and update the text. We corrected the critical radius to cut-off radius to be consistent with the rest of the paper.

We extended the analysis of the population synthesis including a new Figure 7 where we analyze the evolution of the cut-off radius to understand the lifetime of compact discs. We showed that, especially when the external photoevaporation is included, a class of compact discs (with cut-off radii on the order of the gravitational radius) can indeed survive more than 10 Myr.