from py4dgeo.epoch import *
from py4dgeo import *

from vedo import *

PreviousSelection = None

def func(evt): # called every time the mouse moves
    # evt is a dotted dictionary

    if evt.keyPressed == 'z':
        print("reset")
        button.status( button.states )
        #button.switch()
        pass

    if not evt.actor:
        return  # no hit, return
    print("point coords =", evt.picked3d)
    if evt.isPoints:
        print(evt.actor)
    # print("full event dump:", evt)

util.ensure_test_data_availability()
Epoch0, Epoch1 = read_from_xyz("plane_horizontal_t1.xyz", "plane_horizontal_t2.xyz")


plt = Plotter(axes=3)
plt.add_callback('EndInteraction', func)
plt.add_callback('KeyRelease', func)

button = plt.add_button(
    lambda :None,
    pos=(0.9, 0.9),  # x,y fraction from bottom left corner
    states=["Reset", "Epoch A", "Epoch B"],
    c=["g", "r"],
    bc=["dg", "dv"],  # colors of states
    font="courier",   # arial, courier, times
    size=25,
    bold=True,
    italic=False,
)

#plt.show([Points(Epoch0.cloud), Points(Epoch1.cloud)]).close()
plt.show([Points(Epoch0.cloud, c=(1, 0, 1)), Points(Epoch1.cloud, c=(0, 1, 0))]).close()

#plt.show(elli).close()