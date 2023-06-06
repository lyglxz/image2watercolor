# WaterColor
Main Effect of this project is based on [Interactive watercolor rendering with temporal coherence and abstraction](https://dl.acm.org/doi/abs/10.1145/1124728.1124751?casa_token=1-QCDAjlabQAAAAA:8dfN5pI13zgsCcdyYVUtiV5nASf6SMqnzaA3WpwaLY4BK3FAlrmpmoXC2gXvOwk5nWHGLskuaXmjzIc)  
And also, [Towards Photo Watercolorization
with Artistic Verisimilitude](https://ieeexplore.ieee.org/document/6732968#:~:text=Towards%20Photo%20Watercolorization%20with%20Artistic%20Verisimilitude%20Abstract:%20We,paintings%20that%20have%20not%20been%20well%20implemented%20before) inspire me of some edge effects.
## Plan to implement
1. render pipeline implemented in python and cpp.
2. example code for esay use
3. Andorid package
4. Android demo app for esay use
5. pipeline optimize for real-time performance
6. Make All module args configable

## For Python:
quick start:` 
```shell
cd code/python
# render image
python main.py ../../resource/image/input/01.png

# render images in dir 
python main.py ../../resource/image/input

# render with personal config
python main.py ./resource/configs/default.json ../../resource/image/input/01.png
```