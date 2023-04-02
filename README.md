# BABFNet
implementation of BABFNet (TGRS)
implementation of BABFNet (TGRS)
This is a reference implementation of Boundary-Aware Bilateral Fusion Network for Cloud Detection in PyTorch. BABFNet introduces a shallow boundary prediction branch in an encod-decoder-based cloud detection network and contains two key components: SEM and BFM. With the help of multi-stage encoder features, SEM introduces the necessary high-level semantic information to the boundary prediction branch as a supplement to the shallow architecture to improve its boundary prediction performance. At the end of the network, a BFM module with parallel structure is constructed. According to the different properties and target tasks of the two branches, the feature fusion mode is designed respectively to realize effective information interaction. The two promote each other and achieve spiraling performance in the training stage.
Specific implementation details and references are being supplemented...
