# Epilepsy-Detection-using-Convolutional-Neural-Network

For individuals with drug-resistant epilepsy, responsive neurostimulation sys-
tems hold promise for augmenting current therapies and transforming epilepsy
care.
Of the more than two million Americans who suer from recurrent, spon-
taneous epileptic seizures, 500,000 continue to experience seizures despite
multiple attempts to control the seizures with medication. For these pa-
tients responsive neurostimulation represents a possible therapy capable of
aborting seizures before they aect a patient's normal activities. In order for
a responsive neurostimulation device to successfully stop seizures, a seizure
must be detected and electrical stimulation applied as early as possible. A
seizure that builds and generalizes beyond its area of origin will be very dif-
ficult to abort via neurostimulation. Current seizure detection algorithms in
commercial responsive neurostimulation devices are tuned to be hypersensi-
tive, and their high false positive rate results in unnecessary stimulation.
In addition, physicians and researchers working in epilepsy must often
review large quantities of continuous EEG data to identify seizures, which in
some patients may be quite subtle. Automated algorithms to detect seizures
in large EEG datasets with low false positive and false negative rates would
greatly assist clinical care and basic research.
In this project you will be given datasets from patients with epilepsy
undergoing intracranial EEG monitoring to identify a region of brain that
can be resected to prevent future seizures are included in the contest. These
datasets have varying numbers of electrodes and are sampled at 5000 Hz,
with recorded voltages referenced to an electrode outside the brain.
Since the number of electrodes and the recording conditions differ across
patients, you will train a dierent classier for each patient.
