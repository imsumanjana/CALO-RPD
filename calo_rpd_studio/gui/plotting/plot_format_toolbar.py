"""Interactive plot-formatting toolbar for all scientific figures."""
from __future__ import annotations
from copy import deepcopy
import weakref
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget,QGridLayout,QVBoxLayout,QLabel,QComboBox,QFontComboBox,QSpinBox,QDoubleSpinBox,
    QCheckBox,QLineEdit,QPushButton,QFileDialog,QScrollArea
)
from calo_rpd_studio.visualization.plot_style import PlotStyle

TARGETS=("All plot text","Title","X-axis label","Y-axis label","X-axis ticks","Y-axis ticks","Legend","Annotations")
FONT_FIELDS={"Title":"title_font","X-axis label":"x_label_font","Y-axis label":"y_label_font","X-axis ticks":"x_tick_font","Y-axis ticks":"y_tick_font","Legend":"legend_font","Annotations":"annotation_font"}
SIZE_FIELDS={"Title":"title_size","X-axis label":"x_label_size","Y-axis label":"y_label_size","X-axis ticks":"x_tick_size","Y-axis ticks":"y_tick_size","Legend":"legend_size","Annotations":"annotation_size"}

class PlotFormattingToolbar(QWidget):
    def __init__(self,plot_widget,parent=None):
        super().__init__(parent);self.plot_widget_ref=weakref.ref(plot_widget);self.style=deepcopy(plot_widget.style);outer=QVBoxLayout(self);outer.setContentsMargins(0,0,0,0);outer.setSpacing(5)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setMaximumHeight(250);body=QWidget();grid=QGridLayout(body);grid.setContentsMargins(8,6,8,6);grid.setHorizontalSpacing(8);grid.setVerticalSpacing(5);scroll.setWidget(body);outer.addWidget(scroll)
        self.target=QComboBox();self.target.addItems(TARGETS);self.font=QFontComboBox();self.size=QSpinBox();self.size.setRange(6,72);self.bold=QCheckBox("Bold");self.italic=QCheckBox("Italic");self.apply_font_all=QPushButton("Apply font to all text")
        grid.addWidget(QLabel("Text target"),0,0);grid.addWidget(self.target,0,1);grid.addWidget(QLabel("Font"),0,2);grid.addWidget(self.font,0,3);grid.addWidget(QLabel("Size"),0,4);grid.addWidget(self.size,0,5);grid.addWidget(self.bold,0,6);grid.addWidget(self.italic,0,7);grid.addWidget(self.apply_font_all,0,8)
        self.title_text=QLineEdit();self.x_text=QLineEdit();self.y_text=QLineEdit();grid.addWidget(QLabel("Title"),1,0);grid.addWidget(self.title_text,1,1,1,2);grid.addWidget(QLabel("X label"),1,3);grid.addWidget(self.x_text,1,4,1,2);grid.addWidget(QLabel("Y label"),1,6);grid.addWidget(self.y_text,1,7,1,2)
        self.legend_show=QCheckBox("Show legend");self.legend_location=QComboBox();self.legend_location.addItems(["best","upper right","upper left","lower left","lower right","center","center left","center right","upper center","lower center"]);self.legend_columns=QSpinBox();self.legend_columns.setRange(1,12);self.legend_frame=QCheckBox("Legend frame");self.legend_labels=QLineEdit();self.legend_labels.setPlaceholderText("Rename legend: old=new; old2=new2")
        grid.addWidget(self.legend_show,2,0);grid.addWidget(QLabel("Location"),2,1);grid.addWidget(self.legend_location,2,2);grid.addWidget(QLabel("Columns"),2,3);grid.addWidget(self.legend_columns,2,4);grid.addWidget(self.legend_frame,2,5);grid.addWidget(self.legend_labels,2,6,1,3)
        self.x_scale=QComboBox();self.x_scale.addItems(["linear","log"]);self.y_scale=QComboBox();self.y_scale.addItems(["linear","log"]);self.x_min=QLineEdit();self.x_max=QLineEdit();self.y_min=QLineEdit();self.y_max=QLineEdit();self.major_grid=QCheckBox("Major grid");self.minor_grid=QCheckBox("Minor grid");self.axis_width=QDoubleSpinBox();self.axis_width.setRange(.1,10);self.axis_width.setSingleStep(.1)
        grid.addWidget(QLabel("X scale"),3,0);grid.addWidget(self.x_scale,3,1);grid.addWidget(QLabel("Y scale"),3,2);grid.addWidget(self.y_scale,3,3);grid.addWidget(QLabel("X min/max"),3,4);grid.addWidget(self.x_min,3,5);grid.addWidget(self.x_max,3,6);grid.addWidget(QLabel("Y min/max"),3,7);grid.addWidget(self.y_min,3,8);grid.addWidget(self.y_max,3,9);grid.addWidget(self.major_grid,3,10);grid.addWidget(self.minor_grid,3,11);grid.addWidget(QLabel("Axis width"),3,12);grid.addWidget(self.axis_width,3,13)
        self.line_width=QDoubleSpinBox();self.line_width.setRange(.1,12);self.line_width.setSingleStep(.1);self.line_style=QComboBox();self.line_style.addItems(["-","--","-.",":"]);self.marker=QComboBox();self.marker.addItems(["","o","s","^","v","D","x","+","."]);self.marker_size=QDoubleSpinBox();self.marker_size.setRange(0,30);self.series_visible=QCheckBox("Series visible")
        grid.addWidget(QLabel("Line width"),4,0);grid.addWidget(self.line_width,4,1);grid.addWidget(QLabel("Line style"),4,2);grid.addWidget(self.line_style,4,3);grid.addWidget(QLabel("Marker"),4,4);grid.addWidget(self.marker,4,5);grid.addWidget(QLabel("Marker size"),4,6);grid.addWidget(self.marker_size,4,7);grid.addWidget(self.series_visible,4,8)
        self.export_format=QComboBox();self.export_format.addItems(["PNG","SVG","PDF"]);self.dpi=QSpinBox();self.dpi.setRange(72,2400);self.width=QDoubleSpinBox();self.width.setRange(1,30);self.width.setSuffix(" in");self.height=QDoubleSpinBox();self.height.setRange(1,30);self.height.setSuffix(" in");self.transparent=QCheckBox("Transparent");self.tight=QCheckBox("Tight bounding box");self.export_button=QPushButton("Export figure");self.save_style=QPushButton("Save style");self.load_style=QPushButton("Load style");self.reset_style=QPushButton("Reset style");self.apply_all=QPushButton("Apply to all compatible plots")
        grid.addWidget(QLabel("Export"),5,0);grid.addWidget(self.export_format,5,1);grid.addWidget(QLabel("DPI"),5,2);grid.addWidget(self.dpi,5,3);grid.addWidget(QLabel("Width"),5,4);grid.addWidget(self.width,5,5);grid.addWidget(QLabel("Height"),5,6);grid.addWidget(self.height,5,7);grid.addWidget(self.transparent,5,8);grid.addWidget(self.tight,5,9);grid.addWidget(self.export_button,5,10);grid.addWidget(self.save_style,5,11);grid.addWidget(self.load_style,5,12);grid.addWidget(self.reset_style,5,13);grid.addWidget(self.apply_all,5,14)
        self._loading=False;self._connect();self.load_from_style()
    def _connect(self):
        self.target.currentTextChanged.connect(self._sync_target_controls);self.font.currentFontChanged.connect(self._text_changed);self.size.valueChanged.connect(self._text_changed);self.bold.toggled.connect(self._text_changed);self.italic.toggled.connect(self._text_changed);self.apply_font_all.clicked.connect(self._apply_font_all)
        for w in (self.title_text,self.x_text,self.y_text,self.legend_labels,self.x_min,self.x_max,self.y_min,self.y_max):w.editingFinished.connect(self._general_changed)
        for w in (self.legend_show,self.legend_frame,self.major_grid,self.minor_grid,self.series_visible,self.transparent,self.tight):w.toggled.connect(self._general_changed)
        for w in (self.legend_location,self.x_scale,self.y_scale,self.line_style,self.marker):w.currentTextChanged.connect(self._general_changed)
        for w in (self.legend_columns,self.axis_width,self.line_width,self.marker_size):w.valueChanged.connect(self._general_changed)
        self.export_button.clicked.connect(self.export_figure);self.save_style.clicked.connect(self.save_style_profile);self.load_style.clicked.connect(self.load_style_profile);self.reset_style.clicked.connect(self.reset);self.apply_all.clicked.connect(self._apply_all)
    def _set_combo(self,combo,text):
        i=combo.findText(str(text));combo.setCurrentIndex(max(i,0))
    def load_from_style(self):
        self._loading=True;s=self.style;self.title_text.setText(s.title_override);self.x_text.setText(s.x_label_override);self.y_text.setText(s.y_label_override);self.legend_show.setChecked(s.show_legend);self._set_combo(self.legend_location,s.legend_location);self.legend_columns.setValue(s.legend_columns);self.legend_frame.setChecked(s.legend_frame);self.legend_labels.setText('; '.join(f'{k}={v}' for k,v in s.legend_label_overrides.items()));self._set_combo(self.x_scale,s.x_scale);self._set_combo(self.y_scale,s.y_scale)
        for edit,val in ((self.x_min,s.x_min),(self.x_max,s.x_max),(self.y_min,s.y_min),(self.y_max,s.y_max)):edit.setText('' if val is None else str(val))
        self.major_grid.setChecked(s.major_grid);self.minor_grid.setChecked(s.minor_grid);self.axis_width.setValue(s.axis_line_width);self.line_width.setValue(s.line_width);self._set_combo(self.line_style,s.line_style);self._set_combo(self.marker,s.marker);self.marker_size.setValue(s.marker_size);self.series_visible.setChecked(s.series_visible);self.dpi.setValue(300);self.width.setValue(7.2);self.height.setValue(4.6);self.tight.setChecked(True);self._loading=False;self._sync_target_controls()
    def _sync_target_controls(self):
        if self._loading:return
        self._loading=True;target=self.target.currentText();s=self.style
        if target=="All plot text":font=s.title_font;size=s.title_size;bold=s.title_bold;italic=s.title_italic
        else:
            font=getattr(s,FONT_FIELDS[target]);size=getattr(s,SIZE_FIELDS[target]);bold=(s.title_bold if target=="Title" else s.axis_labels_bold if target in ("X-axis label","Y-axis label") else s.tick_labels_bold if "ticks" in target else s.legend_bold if target=="Legend" else s.annotations_bold);italic=(s.title_italic if target=="Title" else s.axis_labels_italic if target in ("X-axis label","Y-axis label") else s.tick_labels_italic if "ticks" in target else s.legend_italic if target=="Legend" else s.annotations_italic)
        self.font.setCurrentFont(QFont(font));self.size.setValue(size);self.bold.setChecked(bold);self.italic.setChecked(italic);self._loading=False
    def _text_changed(self,*_):
        if self._loading:return
        target=self.target.currentText();font=self.font.currentFont().family();size=self.size.value();bold=self.bold.isChecked();italic=self.italic.isChecked();s=self.style
        targets=list(FONT_FIELDS) if target=="All plot text" else [target]
        for t in targets:setattr(s,FONT_FIELDS[t],font);setattr(s,SIZE_FIELDS[t],size)
        if target in ("All plot text","Title"):s.title_bold=bold;s.title_italic=italic
        if target=="All plot text" or target in ("X-axis label","Y-axis label"):s.axis_labels_bold=bold;s.axis_labels_italic=italic
        if target=="All plot text" or "ticks" in target:s.tick_labels_bold=bold;s.tick_labels_italic=italic
        if target in ("All plot text","Legend"):s.legend_bold=bold;s.legend_italic=italic
        if target in ("All plot text","Annotations"):s.annotations_bold=bold;s.annotations_italic=italic
        self._redraw()
    def _apply_font_all(self):
        font=self.font.currentFont().family()
        for field in FONT_FIELDS.values():setattr(self.style,field,font)
        self._redraw()
    def _float_or_none(self,text):
        try:return float(text) if text.strip() else None
        except ValueError:return None
    def _parse_legend_labels(self):
        out={}
        for item in self.legend_labels.text().split(';'):
            if '=' in item:
                old,new=item.split('=',1);out[old.strip()]=new.strip()
        return out
    def _general_changed(self,*_):
        if self._loading:return
        s=self.style;s.title_override=self.title_text.text();s.x_label_override=self.x_text.text();s.y_label_override=self.y_text.text();s.show_legend=self.legend_show.isChecked();s.legend_location=self.legend_location.currentText();s.legend_columns=self.legend_columns.value();s.legend_frame=self.legend_frame.isChecked();s.legend_label_overrides=self._parse_legend_labels();s.x_scale=self.x_scale.currentText();s.y_scale=self.y_scale.currentText();s.x_min=self._float_or_none(self.x_min.text());s.x_max=self._float_or_none(self.x_max.text());s.y_min=self._float_or_none(self.y_min.text());s.y_max=self._float_or_none(self.y_max.text());s.major_grid=self.major_grid.isChecked();s.minor_grid=self.minor_grid.isChecked();s.axis_line_width=self.axis_width.value();s.line_width=self.line_width.value();s.line_style=self.line_style.currentText();s.marker=self.marker.currentText();s.marker_size=self.marker_size.value();s.series_visible=self.series_visible.isChecked();self._redraw()
    def _redraw(self):
        widget=self.plot_widget_ref()
        if widget:widget.apply_style(self.style)
    def _apply_all(self):
        widget=self.plot_widget_ref()
        if widget:widget.manager.apply_to_all(self.style)
    def save_style_profile(self):
        path,_=QFileDialog.getSaveFileName(self,"Save plot style","plot_style.json","JSON (*.json)")
        if path:self.style.save(path)
    def load_style_profile(self):
        path,_=QFileDialog.getOpenFileName(self,"Load plot style","","JSON (*.json)")
        if path:self.style=PlotStyle.load(path);self.load_from_style();self._redraw()
    def reset(self):self.style=PlotStyle();self.load_from_style();self._redraw()
    def export_figure(self):
        fmt=self.export_format.currentText().lower();path,_=QFileDialog.getSaveFileName(self,"Export figure",f"figure.{fmt}",f"{fmt.upper()} (*.{fmt})")
        if path:
            widget=self.plot_widget_ref();widget.manager.export(widget.plot_id,path,self.dpi.value(),self.transparent.isChecked(),self.tight.isChecked(),self.width.value(),self.height.value())
