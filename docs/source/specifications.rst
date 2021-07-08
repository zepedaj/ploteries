Ploteries is a plotting library inspired from tensorboard that aims to be provide greater flexibility, extensibility, and separation of storage and visualization. It provides an extensive suite of visualizations. Ploteries is written entirely in python (based on plotly/Dash), and hence supports easier extensibility by the python community.

Similarly to Tensorboard, Ploteries supports

* Live :ref:`smoothing` of line plots (as well as other plot types).
* The ability to display plots while data is being generated (e.g., during a deep learning training run).
* An API extension that makes it possible to use Ploteries as a drop-in replacement for Tensorboard.
* Data storage in a single file called a ploteries database.
* Support for multiple training runs written to the same ploteries database. 

Besides this, ploteries also provides the following functionality:

* A transparent data storage mechanism that understands :class:`torch.Tensor` or :class:`numpy.ndarray` (including record arrays of all dtypes except `'O'` dtypes). Data added to a plotteries database can be read back as :class:`numpy.ndarray` arrays.
* A flexible visualization engine that is separate from the data storage mechanism and that includes
  * Support for all the plots in the plotly/Dash library.
  * Definition of custom plot types, and the ability to create/delete/edit visualizations at any time from the CLI or Python API.
  * Flexible organization into tabs based on the plot tag or explicit tab assignment.
  * A data explorer that enables browsing through all ploteries databases in a root directory, opening/displaying visualizations from more than one database simultaneously.
  * Documentation on computation of histograms.
  * Optional storage of images and videos as independent media files outside the ploteries database. The root directory of these files can also be specified when the ploteries visualization server is launched, thus making it possible to e.g., avoid sending heavy image datasets to collaborators that already have those images in local storage. Image visualizations can optionally also store a transform definition that can be applied when the visualized image is served, and further support visualization of metadata (e.g., bounding boxes).
* The ability to pause the update of visualizations, or to globally rewind all visualizations to a previous point in time.
* Easier, python-only extensibility.

Under the hood mechanisms:

* Separate queuing thread and writing thread. The queueing thread receives all data write requests and
* Fast update of visualiztions:
  * Hierarchical update that builds a visualization starting from a full, low-resolution plot.
  *
* Single-table stores all data to make it easy to obtain the full range of global steps, and to suppport resolution downsampling, or client-side updating.
* Checksum-based verification of server/client data parity.


 .. todo ::
 * Decouple data store and visualization
   * Create a parent DataStore class with no data_records.index column, and a derived class that has the index column.
   * Change data_records.index to data_records.time_step.
   * Consider adding data record insertion support by means of a "parent" data_records column.
   
