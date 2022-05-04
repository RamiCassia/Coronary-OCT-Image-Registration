
from temporal_registration import *


#### ==== some signal ==== ####
# region
# Example A
# region
N = 150 # total artery length
n1 = 120 # length of first measurement. Should be smaller than N.
n2 = 80 # length of second measurement. Should be smaller than N.
true_shift = 50 # shift of second measurement wrt first measurement. Should be positive and smaller than n1

true_artery_branches = [0] * N  # will contain true artery positions. 
# Next lines populate the list with arbitrary positions. Don't change for now.
branch_ind = list(range(11 ,15)) + list(range(30,40)) + list(range(70,73)) + list(range(90,98)) + list(range(120,125))
for i in branch_ind:
    true_artery_branches[i] = 1

# s1 and s2 create the true artery positions within each measurement area
s1 = true_artery_branches[0:n1]
s2 = true_artery_branches[true_shift:(n2 + true_shift)]

# endregion

# Example B
# region
N = 271 # total artery length
n1 = 200 # length of first measurement. Should be smaller than N.
n2 = 100 # length of second measurement. Should be smaller than N.
true_shift = 50 # shift of second measurement wrt first measurement. Should be positive and smaller than n1

true_artery_branches = [0] * N  # will contain true artery positions. 
# Next lines populate the list with arbitrary positions. Don't change for now.
branch_ind = list(range(40 ,56)) + [113, 114] + list(range(140,160)) + list(range(179,184)) + list(range(223,237)) + list(range(246,257))
for i in branch_ind:
    true_artery_branches[i] = 1

# show the true artery positions
plt.figure()
plt.plot(true_artery_branches)
plt.show()

# s1 and s2 create the true artery positions within each measurement area
s1 = true_artery_branches[0:n1]
s2 = true_artery_branches[true_shift:(n2 + true_shift)]

# endregion

# Example C
# region

n1 = 541 # length of first measurement. Should be smaller than N.
n2 = 541 # length of second measurement. Should be smaller than N.

s1 = [0] * n1
# 084-480-861 BL: (n = 541)
branch_ind_1 = list(range(86 ,96)) + list(range(100,133)) + list(range(151,156)) + list(range(191,204)) + list(range(227,238)) + list(range(329,347)) + list(range(354,364)) + list(range(402,413)) + list(range(430,437)) + list(range(445,457))
for i in branch_ind_1:
    s1[i] = 1

s2 = [0] * n2
# 084-480-861 FUP: (n = 541)
branch_ind_2 = [14, 15, 16] + list(range(18 ,36)) + list(range(108,159)) + list(range(185,192)) + list(range(216,235)) + list(range(272,276)) + list(range(353,385)) + list(range(387,395)) + list(range(399,412)) + list(range(429,440)) + list(range(454,464))+ list(range(468,478)) + [480, 481]
for i in branch_ind_2:
    s2[i] = 1

# endregion

# Example D
# region

n1 = 271 # length of first measurement. Should be smaller than N.
n2 = 271 # length of second measurement. Should be smaller than N.

s1 = [0] * n1
# 453-636-202 BL: (n = 271)
branch_ind_1 = [0, 1, 2, 3, 18, 27] + list(range(41 ,53)) + [77, 89, 99] + list(range(140,192)) + [212] + list(range(214,218)) + [222] + list(range(224,231))
for i in branch_ind_1:
    s1[i] = 1

s2 = [0] * n2
# 453-636-202 FUP: (n = 271)
branch_ind_2 = list(range(22 ,27)) + list(range(59,64)) + list(range(82,88)) + list(range(118,130)) + list(range(155,171)) + list(range(174,180)) + list(range(181,193))
for i in branch_ind_2:
    s2[i] = 1

# endregion

# Example E
# region

n1 = 271 # length of first measurement. Should be smaller than N.
n2 = 271 # length of second measurement. Should be smaller than N.

s1 = [0] * n1
# 017-750-526 BL: (n = 271)
branch_ind_1 = list(range(37 ,44)) + list(range(108,112)) + list(range(178,183)) + list(range(225,235)) + list(range(245,249))
for i in branch_ind_1:
    s1[i] = 1

s2 = [0] * n2
# 017-750-526 FUP: (n = 271)
branch_ind_2 = list(range(40 ,56)) + [113, 114] + list(range(140,160)) + list(range(179,184)) + list(range(223,237)) + list(range(246,257))
for i in branch_ind_2:
    s2[i] = 1

# endregion


# endregion


#### === visualize signal and add noise to signal in order to get some mock data ==== ####
# region

n1 = len(s1)
n2 = len(s2)

# show arteries in each section
plt.figure()
plt.plot(range(max(len(s1), len(s2))), s1, s2)
plt.show()

# create noisy data
t1, t2 = add_beta_noise(s1, s2, c = 0.05)

# visualize the kind of data the neural network might output
plt.figure()
plt.plot(range(max(len(t1), len(t2))), t1, t2)
plt.show()

# endregion





#### ==== apply temporal_registration to mock data ==== ####
# region

# Minimal overlap recommendation:
minimal_overlap_edge_branches(t1, t2)
minimal_overlap_edge_branches(t1, t2, cutoff = 0.5, min_branch_size=5)



# Example Application of temporal_registration
res = temporal_registration(t1 = t1, t2 = t2, minimal_overlap = "auto", step_pattern = "symmetricP2", smoother="moving_median", smoothing_window=7, discr_cutoff=0.9, min_branch = 3)
shift, t1_start, t1_end, t2_start, t2_end = res[0]
t1_trafo = res[1]
t2_trafo = res[2]
t1_overlap = t1_trafo[t1_start:(t1_end + 1)]
t2_overlap = t2_trafo[t2_start:(t2_end+1)]
alignment = dtw.dtw(t1_overlap, t2_overlap, keep_internals=True, step_pattern = "symmetricP2")
alignment.plot(type="twoway",offset=-2) # might give an error, not sure why
alignment.plot()

res[4]
t1_start + alignment.index1
t2_start + alignment.index2

# Visualize Results
offset1 = -1.5
offset2 = -3

t1_trafo = res[1]
t2_trafo = res[2]

plt.figure()
plt.plot(range(max(len(t1), len(t2))), t1, [t2[j] + offset1 for j in range(len(t2))])
plt.plot(range(max(len(t1), len(t2))), t1_trafo, [t2_trafo[j] + offset1 for j in range(len(t2_trafo))], color = "black", linestyle = "dashed")
plt.show()

alignment.index1
alignment.index2
wr = dtw.warp(alignment, index_reference=True)
wq = dtw.warp(alignment, index_reference=False)

warped_t2 = [t2_overlap[j] + offset2 for j in wr]
len(wr)
len(t2_overlap)
len(warped_t2)


plt.figure()
plt.axvline(x = t1_start, color = "gray", linestyle = "dashed")
plt.axvline(x = t1_end, color = "gray", linestyle = "dashed")
plt.plot(t1)
plt.plot([x + shift for x in range(len(t2))], [t2[j] + offset1 for j in range(len(t2))])
plt.plot([x + shift + t2_start for x in range(len(warped_t2))], warped_t2)
plt.show()



'''
# We need to determine the value of the function parameter "minimal_overlap"
# One approach: use different values and aim for a output-overlap a good bit larger than the specified minimal_overlap
min_overlap_list = [20, 30, 40, 50, 60]
overlap_output = [None] * 5
for i in range(len(min_overlap_list)):
    overlap, t1_shift, t2_shift = temporal_registration(t1 = t1, t2 = t2, minimal_overlap = min_overlap_list[i])
    overlap_output[i] = overlap

plt.figure()
plt.plot(min_overlap_list, overlap_output)
plt.scatter(min_overlap_list, overlap_output)
plt.show()

'''



# endregion

