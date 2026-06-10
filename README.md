# AMD Strix Halo ROCm GPU Memory Debug

## Débloquer la mémoire GPU utilisable sur AMD Strix Halo gfx1151 sous ROCm / PyTorch

Ce repository documente un diagnostic réel effectué sur un mini-PC AMD Strix Halo avec ROCm et PyTorch.

Objectif : comprendre pourquoi la mémoire GPU utilisable était limitée à environ 16 Go, puis obtenir environ 64 Go exploitables sans changer de matériel.

---

## Résultat

| Élément                        |                  Avant |                  Après |
| ------------------------------ | ---------------------: | ---------------------: |
| UMA Frame Buffer BIOS          |                  96 Go |                   2 Go |
| Mémoire GPU utilisable PyTorch |                 ~16 Go |                 ~64 Go |
| Plateforme                     | AMD Strix Halo gfx1151 | AMD Strix Halo gfx1151 |
| Stack                          |         ROCm / PyTorch |         ROCm / PyTorch |

Le gain principal vient d’un réglage BIOS : réduire l’UMA Frame Buffer au minimum afin de laisser le GTT gérer la mémoire utilisée par ROCm/HIP.

---

## Environnement testé

Machine :

* GMKtec NucBox EVO-X2
* Carte : AXB35-02
* APU : AMD Ryzen AI MAX+ 395
* GPU intégré : Radeon 8060S
* Architecture GPU : gfx1151
* RAM : 128 Go mémoire unifiée
* BIOS : EVO-X2 1.12

Système :

* Ubuntu 24.04
* Kernel : 6.17.0-1025-oem
* ROCm 7.1
* PyTorch : 2.13.0.dev+rocm7.1

---

## Problèmes rencontrés

Deux problèmes distincts ont été observés.

### 1. Blocage GPU au premier calcul

Symptôme :

* PyTorch détecte bien le GPU
* `torch.cuda.is_available()` retourne `True`
* mais la première opération GPU bloque
* pas d’exception Python claire
* logs kernel avec page fault gfxhub

Cause probable :

* firmware AMD trop ancien pour gfx1151
* version MES problématique

Correctif :

* mise à jour du firmware AMD depuis `linux-firmware`
* mise à jour initramfs
* reboot

Résultat :

* les calculs GPU fonctionnent
* les tests PyTorch passent
* performance bf16 mesurée : environ 31,9 TFLOP/s

Important : ce correctif règle le blocage GPU, mais il ne règle pas la quantité de mémoire disponible.

---

### 2. Mémoire GPU plafonnée à environ 16 Go

Symptôme :

```bash
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory/1e9)"
```

Résultat observé :

```text
16.6
```

Autre symptôme :

```bash
free -g
```

Résultat observé :

```text
total 30
```

La machine possède pourtant 128 Go de mémoire unifiée.

Diagnostic kernel :

```bash
sudo dmesg | grep -iE "amdgpu: (VRAM|GTT|GART)"
```

Observation :

```text
VRAM: 98304M
GART: 512M
```

Le BIOS réservait environ 96 Go en UMA fixe.

---

## Cause racine

Sur cet APU, ROCm/HIP n’utilise pas simplement le gros bloc UMA réservé par le BIOS pour la mémoire de calcul.

La mémoire exploitable par PyTorch/ROCm dépend fortement du GTT, c’est-à-dire de la RAM système mappée dynamiquement pour le GPU.

En réservant 96 Go en UMA fixe, le BIOS réduit fortement la RAM système disponible. Résultat : le GTT devient trop limité et PyTorch ne peut utiliser qu’environ 16 Go.

Conclusion contre-intuitive :

```text
Un gros UMA BIOS peut réduire la mémoire réellement utilisable par ROCm.
```

---

## Correctif appliqué

Action réalisée dans le BIOS :

```text
UMA Frame Buffer Size : 2 Go
```

Sur cette machine, 2 Go était la valeur minimale proposée.

Aucun changement logiciel n’a été nécessaire pour passer de 16 Go à environ 64 Go utilisables.

---

## Vérification après correction

Test PyTorch :

```python
import torch

last = 0

try:
    for gb in [8, 16, 24, 32, 48, 64, 80]:
        x = torch.empty(int(gb * 1e9 / 2), dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        last = gb
        print(f"{gb} Go : OK")
        del x
        torch.cuda.empty_cache()

except RuntimeError as e:
    print(f"échec après {last} Go ->", str(e)[:90])
```

Résultat obtenu :

```text
8 Go : OK
16 Go : OK
24 Go : OK
32 Go : OK
48 Go : OK
64 Go : OK
échec à 80 Go
```

La mémoire utilisable passe donc d’environ 16 Go à environ 64 Go.

---

## Pourquoi environ 64 Go et pas 128 Go ?

Le kernel dimensionne généralement le GTT autour d’une partie de la RAM système disponible.

Sur une machine avec 128 Go de RAM, obtenir environ 64 Go utilisables côté PyTorch/ROCm est cohérent.

Ce n’est pas forcément une erreur.

---

## Variables d’environnement utilisées

Les variables suivantes étaient déjà utilisées avant et après la correction :

```bash
export HSA_XNACK=1
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export HIP_VISIBLE_DEVICES=0
export ROCBLAS_USE_HIPBLASLT=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Elles aident PyTorch/ROCm à fonctionner correctement avec cette plateforme, mais elles ne sont pas la cause du passage de 16 Go à 64 Go.

Le changement déterminant reste le réglage BIOS UMA.

---

## Firmware fix

Commande utilisée pour mettre à jour le firmware AMD :

```bash
cd /tmp
git clone --depth 1 https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git
sudo mkdir -p /lib/firmware/updates/amdgpu
sudo cp /tmp/linux-firmware/amdgpu/gc_11_5* /lib/firmware/updates/amdgpu/
sudo update-initramfs -u -k all
sudo reboot
```

Vérification :

```bash
sudo sh -c 'cat /sys/kernel/debug/dri/*/amdgpu_firmware_info' | grep -i "MES "
```

Avant :

```text
firmware version: 0x00000083
```

Après :

```text
firmware version: 0x00000088
```

---

## Points importants

* Le firmware corrige le blocage GPU.
* Le BIOS UMA corrige la mémoire utilisable.
* Les deux sujets ne doivent pas être confondus.
* La mémoire GPU utilisable par ROCm dépend fortement du GTT.
* Réduire l’UMA BIOS peut améliorer la mémoire disponible pour PyTorch.
* Le résultat obtenu sur cette machine est environ 64 Go utilisables.

---

## TL;DR

Sur AMD Strix Halo gfx1151 avec ROCm/PyTorch :

```text
Ne pas réserver trop de mémoire UMA dans le BIOS.
Réduire UMA Frame Buffer Size au minimum.
Laisser ROCm/HIP utiliser le GTT.
```

Résultat obtenu :

```text
~16 Go utilisables avant
~64 Go utilisables après
```

---

## Ce que ce projet démontre

Ce repository montre une capacité à :

* diagnostiquer un problème Linux/GPU
* lire les logs kernel
* comprendre l’impact du BIOS sur ROCm
* distinguer firmware, driver, kernel et runtime PyTorch
* tester proprement une hypothèse
* documenter une résolution technique réelle
