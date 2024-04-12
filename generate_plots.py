import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib
import numpy as np
import pickle

matplotlib.rcParams.update({'font.size': 16})

f = open('test.pkl','rb')
rwmh_means,rwmh_lowers,rwmh_uppers,mala_means,mala_lowers,mala_uppers,hmc_means,hmc_lowers,hmc_uppers,dims = pickle.load(f)
f.close()

xseq = np.linspace(2,2*len(rwmh_means),num=100)
dims = np.array(dims)

blue_patch = mp.Patch(color='blue', label='RWMH')
red_patch = mp.Patch(color='red', label='HMC')
green_patch = mp.Patch(color="green", label="MALA")

fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
plt.setp((ax1,ax2,ax3), xticks=dims)
plt.tight_layout(pad=2)

rwmh_b1, rwmh_b0 = np.polyfit(dims, rwmh_means, deg=1)
ax1.fill_between(x=dims,y1=rwmh_lowers,y2=rwmh_uppers,color="blue",alpha=0.5)
ax1.plot(dims, rwmh_means, color = 'black', marker='o')
ax1.plot(xseq, rwmh_b0 + rwmh_b1*xseq, '--k')
ax1.text(x=0.01,y=0.95,s="RWMH slope: " + str(round(rwmh_b1,5)), transform = ax1.transAxes)
ax1.set_xlabel('d')
ax1.set_ylabel('$\epsilon$')
ax1.set_title("RWMH")
ax1.set_axisbelow(True)
ax1.yaxis.grid(color='gray', linestyle='dashed')
ax1.xaxis.grid(color='gray',linestyle='dashed')
print(rwmh_b1)

mala_b1, mala_b0 = np.polyfit(dims, mala_means, deg=1)
ax2.fill_between(x=dims,y1=mala_lowers,y2=mala_uppers,color="green",alpha=0.5)
ax2.plot(dims, mala_means, color = 'black',marker='o')
ax2.plot(xseq,mala_b0 + mala_b1*xseq, '--k')
ax2.text(x=0.01,y=0.95,s="MALA slope: " + str(round(mala_b1,5)), transform = ax2.transAxes)
ax2.set_xlabel('d')
ax2.set_ylabel('$\epsilon$')
ax2.set_title("MALA")
ax2.set_axisbelow(True)
ax2.yaxis.grid(color='gray', linestyle='dashed')
ax2.xaxis.grid(color='gray',linestyle='dashed')
print(mala_b1)

hmc_b1, hmc_b0 = np.polyfit(dims, hmc_means, deg=1)
ax3.fill_between(x=dims,y1=hmc_lowers,y2=hmc_uppers,color="red",alpha=0.5)
ax3.plot(dims, hmc_means, color = 'black',marker='o')
ax3.plot(xseq, hmc_b0+hmc_b1*(xseq), '--k')
ax3.text(x=0.01,y=0.95,s="HMC slope: " + str(round(hmc_b1,5)), transform = ax3.transAxes)
ax3.set_xlabel('d')
ax3.set_ylabel('$\epsilon$')
ax3.set_title("HMC")
ax3.set_axisbelow(True)
ax3.yaxis.grid(color='gray', linestyle='dashed')
ax3.xaxis.grid(color='gray',linestyle='dashed')
print(hmc_b1)

fig1.savefig("test1.png")

fig2, ax = plt.subplots()
plt.setp(ax, xticks=dims)
ax.fill_between(x=dims, y1=rwmh_lowers,y2=rwmh_uppers,color="blue",alpha=0.5)
ax.fill_between(x=dims, y1=mala_lowers,y2=mala_uppers,color="green",alpha=0.5)
ax.fill_between(x=dims, y1=hmc_lowers,y2=hmc_uppers,color="red",alpha=0.5)
ax.plot(dims, rwmh_means, color = 'black',marker='o', markerfacecolor='blue')
ax.plot(dims, mala_means, color='black',marker='^',markerfacecolor='green')
ax.plot(dims, hmc_means, color="black",marker='s',markerfacecolor='red')
ax.legend(handles=[blue_patch,green_patch,red_patch],loc="upper left")
ax.set_title("Comparison of Methods")
ax.set_xlabel('d')
ax.set_ylabel('$\epsilon$')
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray',linestyle='dashed')

fig2.savefig("test2.png")