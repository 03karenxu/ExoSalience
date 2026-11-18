# CNN model decisions

Why the model in `cnn.py` and its training in `train.py` are set up the way they are. "Untested" means the reasoning is sound but no experiment on this data has confirmed it yet.

| Decision | Choice | Justification |
| --- | --- | --- |
| Two input views | Global view (2001 points, whole orbit) + local view (201 points, ±4 transit durations) | The global view shows context such as a secondary eclipse halfway round the orbit (a sign of an eclipsing binary); the local view shows the dip's shape (U for a planet, V for a grazing binary). Same setup as Astronet (Shallue & Vanderburg, 2018). |
| Separate branch per view | `enc_g` and `enc_l` with their own weights | The two views have different lengths and show different things, so each gets its own pattern detectors. |
| 1D convolutions | `nn.Conv1d` | A light curve is a 1D signal. Convolutions reuse the same small filter at every position, so a dip is recognised wherever it falls, with far fewer weights than a dense layer on raw points. |
| Conv blocks | `conv_block`: conv, ReLU, conv, ReLU, max pool | The same recipe repeats six times (4 global, 2 local), so it is written once. Depth becomes a setting (`channels`) instead of copied layer code. |
| Two convs per block | Two `Conv1d` before each pool | Each extra conv widens what one feature can see before the pool discards detail. Standard in Astronet and VGG-style networks. |
| ReLU after each conv | `nn.ReLU` | Without a non-linearity, stacked convs collapse into one linear filter and gain nothing. ReLU keeps only "pattern present" responses. |
| Max pooling | `MaxPool1d(kernel_size=2, stride=2)` after each block | Halves the length, so the next block's kernel covers twice as much of the original curve: the window grows from 9 points to about 136 by the last global conv. Keeps the strongest response in each region. |
| Kernel size | 5, both branches | Small kernels plus pooling widen the view more cheaply than large kernels. Odd, so each filter stays centred with `padding = k // 2`. |
| Channels double per block | Global 16 → 32 → 64 → 128; local 16 → 32 | Each pool halves the length, so doubling channels keeps 16,000 values at every global stage. Later layers detect combinations of earlier patterns and need more filters. Cost per block stays even. |
| Fewer local blocks | 4 global, 2 local | The local view is 10× shorter; 2 blocks shrink it to 50 positions, similar in scale to the global branch's 125. |
| Flatten instead of global average pooling | `nn.Flatten` | The original model averaged each channel to one number, losing where patterns were. Train and val loss ended equal (0.443 / 0.445) at 0.95 AUC: the model was limited by what it could see, not overfitting. Flattening gives the head a separate weight per position. Untested on this data. |
| Hidden layer in the head | `LazyLinear(256)` + ReLU before the output | One linear layer can only add up evidence; a hidden layer can learn combinations like "dip here and no dip there". |
| Dropout | 0.3, in the head only | The first head layer holds about 4.5M of the model's 4.68M weights. With about 10k training examples, dropout stops it relying on a few units and memorising. |
| LazyLinear + dry run | `LazyLinear` sizes itself; `train.py` runs one batch before creating the optimiser | The flattened size (17,600) depends on input lengths and channels; computing it by hand breaks whenever they change. PyTorch recommends initialising lazy layers before the optimiser collects the weights. |
| Output | One logit; sigmoid applied in the loss and in `evaluate()` | `BCEWithLogitsLoss` is more numerically stable than a sigmoid followed by plain BCE. |
| Loss weighting | `pos_weight` = negatives / positives in the training set (about 3.5 for DR24) | Mild imbalance (22% planets). Weighting improves recall at the default cutoff; resampling would discard data or repeat examples. Makes output probabilities run high, so they are not calibrated. |
| Optimiser | Adam, lr 1e-3, weight decay 1e-5 | Standard defaults that work without tuning; light weight decay discourages large weights. |
| Gradient clipping | `clip_grad_norm_`, max norm 5.0 | Guards against occasional huge gradient steps destabilising training. |
| Batch size | 64 | Common default; small enough for frequent updates, large enough for stable gradients. Untested alternatives. |
| Epochs | 30 | The original model's validation AUC levelled off by about epoch 15; 30 gives room without wasting time (2 s per epoch on a T4). |
| Validation set | 10% of training stars, seed 42, split by star | Picks the best epoch and the cutoff without touching the test set. Split by star so a star's signals never appear in both. |
| Keep the best epoch | Weights from the epoch with the highest validation ROC-AUC | Simple early stopping: guards against overfitting in later epochs. ROC-AUC does not depend on the cutoff. |
| Cutoff | Threshold with the highest validation F1, saved in the checkpoint config | With `pos_weight` the outputs are not calibrated, so 0.5 is arbitrary. F1 weighs precision and recall equally; switch to a recall target if missed planets matter more. |
| Main metric | ROC-AUC, with precision, recall, F1 reported | Accuracy misleads at 22–24% positives (always "no" scores 0.76). ROC-AUC measures ranking independent of the cutoff, so it is the fair way to compare runs. |
| Fixed seeds | `torch.manual_seed(42)`, `np.random.seed(42)` | Makes runs repeatable so settings can be compared. GPU operations can still vary slightly between runs. |
| Tabular input | `x_tab` argument kept but unused | Leaves room to add star properties (radius, temperature) later without changing the model's call signature or `infer.py`. |
