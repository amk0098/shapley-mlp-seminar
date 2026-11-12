# Part III — Discussion {-}

# 15 Discussion and limitations

## 15.1 What the case study shows

1. **Exact before approximate.** With seven features, exact Shapley values cost seconds. They turned the
   comparison of KernelSHAP and DeepSHAP from a matter of belief into a measurement: KernelSHAP is exact with its
   defaults, DeepSHAP is a different additive attribution with a correct sum. Whenever the number of features
   permits, the exact computation should be the reference.
2. **"Approximation" depends on the setting, not on the name.** KernelSHAP becomes an approximation only when
   its budget does not cover all coalitions (here from $M = 12$); then too few coalitions make it unstable.
   DeepSHAP is exact where the network is linear between reference and instance and approximate where ReLUs
   switch.
3. **The question is set by the background, not by the method.** Changing the background changed one
   attribution from +5.90 to +2.47; changing the method (DeepSHAP instead of exact) changed it to +6.07.
   Random 100-day backgrounds vary by 11.5 %, more than the mean DeepSHAP error of 6.5 %.
4. **Runtime is about the number of features.** For small models the overhead of the explainer matters more than
   its asymptotic cost; the exponential growth of exact computation is real but starts to matter only at about a
   dozen features.
5. **Embeddings can be read** if both readings are combined. Reading (b) says which neurons carry the decision,
   reading (a) what these neurons react to; the linear head makes the two consistent with the logit
   explanation.

## 15.2 Limitations of Shapley explanations

**Attribution is not causation.** Shapley values describe how the fitted *model* reacts relative to a
background. A large value for vapour pressure does not mean that vapour pressure causes heat; the false alarm of
Section 11.4 shows that the explanation can be perfectly faithful to the model while the model misses the
physics. Kumar et al. discuss in detail why Shapley-based feature importance does not, by itself, provide the
explanations users expect [Kum20].

**Correlated features.** The interventional value function evaluates the model on inputs that cannot occur
(Exercise 10), and it splits the effect of "temperature" over five features, sometimes with opposite signs. The
values are correct statements about the model, but partly about regions where it was never trained. Remedies are
group attributions (Section 11.5) or conditional Shapley values [AJL21], which in turn may credit features the
model ignores [SN20, JMB20].

**Dependence on the background.** Every explanation answers "compared to which days?". Reported values must name
the background (source, size, seed); different, equally reasonable backgrounds give different numbers.

**Local is not global.** A Shapley value is one number for one prediction. Global rankings such as mean
$|\varphi|$ are summaries built on top of them; they hide signs and interactions, and the summary used must be
named.

**Probability scale.** All values are in logit units. They cannot be converted into percentage points
(Exercise 2), which makes them harder to communicate to a non-technical audience.

## 15.3 Limitations of the case study

- One network, one seed, five stations. Whether the neuron roles of Section 12 or the feature ranking are stable
  across seeds, stations or climates was not tested.
- The features describe only the current and previous days; weather changes (fronts, pressure, wind) are
  invisible to the model. Its errors are therefore partly unexplainable from its inputs.
- The predicted probabilities are not calibrated (class weight, threshold 0.9).
- Runtime results depend on hardware and on the implementation; experiment C uses untrained random networks.
- The conditional value function was discussed but not computed.

## 15.4 Common misconceptions

| Misconception | Why it is wrong | Better |
|---|---|---|
| "A high Shapley value means the feature causes heat days." | It explains the model's output relative to a background. | "The model relies strongly on this feature for this day." |
| "Shapley value = feature importance." | It is local: one instance, one sign. | Separate local attribution and global summary; name the summary. |
| "+1 means +x percentage points." | Additivity holds only on the logit scale. | State the unit: logit (log-odds). |
| "The base value is the prediction for the average day." | It is the average prediction over the background. | "Base value = mean logit over the background." |
| "The background does not matter." | It defines the game; here it changes values by a factor of two. | Fix source, size and seed; report them; show sensitivity. |
| "SHAP is one algorithm." | It is a family with different assumptions and errors. | Always name the explainer and its settings. |
| "KernelSHAP approximates, DeepSHAP is exact because it is specialised." | Here it is the other way round. | Check both against exact values when possible. |
| "Efficiency holds, so the values are correct." | Only the sum is checked (Exercise 16). | Use efficiency as a bug check. |
| "$\varphi_i = 0$: the model does not use feature $i$." | It can be zero for this instance and large elsewhere, or contributions can cancel. | "For this day the feature does not move the prediction away from the base value." |
| "A negative value means a bad feature." | It means: pushes this logit below the base value. | Explain the sign by the class it favours. |

# 16 Conclusion

The question of this paper was how the predictions of an MLP can be explained with Shapley values, and where the
limits of these explanations lie.

**How.** Shapley values distribute the difference between a prediction and the average prediction over the
features, in the unique way that satisfies efficiency, symmetry, dummy and additivity. For a neural network one
must first define what an absent feature means; the interventional value function replaces it by background
data. The values can then be computed exactly for few features, estimated by KernelSHAP through a weighted
regression over coalitions, or approximated by DeepSHAP through a backward pass with the DeepLIFT rules. On the
logit they explain the decision in terms of the input features; on the embedding they show which internal
neurons carry the decision and what these neurons respond to.

**Limits.** The explanation is only as meaningful as the game behind it: it depends on the background, it
evaluates the model on impossible inputs when features are correlated, and it describes the model, not the
world. Approximations add their own error — in this case study smaller than the effect of the background.

For the heat-day network the answer is concrete: the model predicts heat from today's temperatures, vapour
pressure and humidity, mainly by switching off neurons that represent normal weather, and it fails where the
weather changes in ways its inputs cannot see. A natural next step would be to compare conditional and
interventional values on the correlated temperature features, and to check how stable the explanations are
across training seeds.

# References {-}

**[AJL21]** Aas, K.; Jullum, M.; Løland, A.: Explaining individual predictions when features are dependent:
More accurate approximations to Shapley values. *Artificial Intelligence* 298 (2021), 103502.
https://doi.org/10.1016/j.artint.2021.103502

**[DWD]** Deutscher Wetterdienst, Climate Data Center: Daily station observations (climate, "kl").
https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/

**[GBC16]** Goodfellow, I.; Bengio, Y.; Courville, A.: *Deep Learning*. MIT Press, 2016.

**[JMB20]** Janzing, D.; Minorics, L.; Blöbaum, P.: Feature relevance quantification in explainable AI: A causal
problem. In: *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics
(AISTATS)*, PMLR 108, 2020, pp. 2907–2916.

**[Kum20]** Kumar, I. E.; Venkatasubramanian, S.; Scheidegger, C.; Friedler, S.: Problems with
Shapley-value-based explanations as feature importance measures. In: *Proceedings of the 37th International
Conference on Machine Learning (ICML)*, PMLR 119, 2020, pp. 5491–5500.

**[LL17]** Lundberg, S. M.; Lee, S.-I.: A Unified Approach to Interpreting Model Predictions. In: *Advances in
Neural Information Processing Systems 30 (NIPS 2017)*, pp. 4765–4774.

**[Mol22]** Molnar, C.: *Interpretable Machine Learning*. 2nd ed., 2022.
https://christophm.github.io/interpretable-ml-book/

**[Pas19]** Paszke, A. et al.: PyTorch: An Imperative Style, High-Performance Deep Learning Library. In:
*Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*.

**[Sha53]** Shapley, L. S.: A Value for n-Person Games. In: Kuhn, H. W.; Tucker, A. W. (eds.): *Contributions
to the Theory of Games II*. Princeton University Press, 1953, pp. 307–317.

**[SGK17]** Shrikumar, A.; Greenside, P.; Kundaje, A.: Learning Important Features Through Propagating
Activation Differences. In: *Proceedings of the 34th International Conference on Machine Learning (ICML)*,
PMLR 70, 2017, pp. 3145–3153.

**[SN20]** Sundararajan, M.; Najmi, A.: The Many Shapley Values for Model Explanation. In: *Proceedings of the
37th International Conference on Machine Learning (ICML)*, PMLR 119, 2020, pp. 9269–9278.

**[ŠK14]** Štrumbelj, E.; Kononenko, I.: Explaining prediction models and individual predictions with feature
contributions. *Knowledge and Information Systems* 41(3) (2014), pp. 647–665.

**[You85]** Young, H. P.: Monotonic solutions of cooperative games. *International Journal of Game Theory*
14(2) (1985), pp. 65–72.

**Software.** shap 0.54.1 (https://shap.readthedocs.io), PyTorch 2.15.0, scikit-learn 1.9.1, Python 3.13.
**Code and results.** https://github.com/amk0098/heatrisk-shap-analysis (tag `v1.0-talk`).
