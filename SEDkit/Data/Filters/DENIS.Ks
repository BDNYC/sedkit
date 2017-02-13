<?xml version="1.0"?>
<VOTABLE version="1.1" xsi:schemaLocation="http://www.ivoa.net/xml/VOTable/v1.1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <INFO name="QUERY_STATUS" value="OK"/>
  <RESOURCE type="results">
    <TABLE utype="photdm:PhotometryFilter.transmissionCurve.spectrum">
    <PARAM name="FilterProfileService" value="ivo://svo/fps" ucd="meta.ref.ivorn" utype="PhotometryFilter.fpsIdentifier" datatype="char" arraysize="*"/>
    <PARAM name="filterID" value="DENIS/DENIS.Ks" ucd="meta.id" utype="photdm:PhotometryFilter.identifier" datatype="char" arraysize="*"/>
    <PARAM name="WavelengthUnit" value="Angstrom" ucd="meta.unit" utype="PhotometryFilter.SpectralAxis.unit" datatype="char" arraysize="*"/>
    <PARAM name="WavelengthUCD" value="em.wl" ucd="meta.ucd" utype="PhotometryFilter.SpectralAxis.UCD" datatype="char" arraysize="*"/>
    <PARAM name="Description" value="DENIS Ks" ucd="meta.note" utype="photdm:PhotometryFilter.description" datatype="char" arraysize="*"/>
    <PARAM name="PhotSystem" value="DENIS" utype="photdm:PhotometricSystem.description" datatype="char" arraysize="*">
       <DESCRIPTION>Photometric system</DESCRIPTION>
    </PARAM>
    <PARAM name="Band" value="Ks" utype="photdm:PhotometryFilter.bandName" datatype="char" arraysize="*"/>
    <PARAM name="ProfileReference" value="http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_148/index_148.html" datatype="char" arraysize="*"/>
    <PARAM name="Description" value="DENIS Ks" ucd="meta.note" utype="photdm:PhotometryFilter.description" datatype="char" arraysize="*"/>
    <PARAM name="WavelengthMean" value="21626.3400678" unit="Angstrom" ucd="em.wl" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Location.Value" datatype="float" >
       <DESCRIPTION>Mean wavelength. Defined as integ[x*filter(x) dx]/integ[filter(x) dx]</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthEff" value="21465.009653" unit="Angstrom" ucd="em.wl.effective" datatype="float" >
       <DESCRIPTION>Effective wavelength. Defined as integ[x*filter(x)*vega(x) dx]/integ[filter(x)*vega(x) dx]</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthMin" value="19473.5714286" unit="Angstrom" ucd="em.wl;stat.min" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Start" datatype="float" >
       <DESCRIPTION>Minimum filter wavelength. Defined as the first lambda value with a transmission at least 1% of maximum transmission</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthMax" value="23978.75" unit="Angstrom" ucd="em.wl;stat.max" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Stop" datatype="float" >
       <DESCRIPTION>Maximum filter wavelength. Defined as the last lambda value with a transmission at least 1% of maximum transmission</DESCRIPTION>
    </PARAM>
    <PARAM name="WidthEff" value="3001.53230706" unit="Angstrom" ucd="instr.bandwidth" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Extent" datatype="float" >
       <DESCRIPTION>Effective width. Defined as integ[x*filter(x) dx].\nEquivalent to the horizontal size of a rectangle with height equal to maximum transmission and with the same area that the one covered by the filter transmission curve.</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthCen" value="21512.880522" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Central wavelength. Defined as the central wavelength between the two points defining FWMH</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthPivot" value="21604.7531453" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Peak wavelength. Defined as sqrt{integ[x*filter(x) dx]/integ[filter(x) dx/x]}</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthPeak" value="22190" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Peak wavelength. Defined as the lambda value with larger transmission</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthPhot" value="21508.3379976" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Photon distribution based effective wavelength. Defined as integ[x^2*filter(x)*vega(x) dx]/integ[x*filter(x)*vega(x) dx]</DESCRIPTION>
    </PARAM>
    <PARAM name="FWHM" value="3301.94081932" unit="Angstrom" ucd="instr.bandwidth" datatype="float" >
       <DESCRIPTION>Full width at half maximum. Defined as the difference between the two wavelengths for which filter transmission is half maximum</DESCRIPTION>
    </PARAM>
    <PARAM name="PhotCalID" value="DENIS/DENIS.Ks/Vega" ucd="meta.id" utype="photdm:PhotCal.identifier" datatype="char" arraysize="*"/>
    <PARAM name="MagSys" value="Vega" ucd="meta.code" utype="photdm:PhotCal.MagnitudeSystem.type" datatype="char" arraysize="*"/>
    <PARAM name="ZeroPoint" value="667.364186928" unit="Jy" ucd="phot.flux.density" utype="photdm:PhotCal.ZeroPoint.Flux.value" datatype="float" />
    <PARAM name="ZeroPointUnit" value="Jy" ucd="meta.unit" utype="photdm:PhotCal.ZeroPoint.Flux.unit" datatype="char" arraysize="*"/>
    <PARAM name="ZeroPointType" value="Pogson" ucd="meta.code" utype="photdm:PhotCal.ZeroPoint.type" datatype="char" arraysize="*"/>
      <FIELD name="Wavelength" utype="spec:Data.SpectralAxis.Value" ucd="em.wl" unit="Angstrom" datatype="float"/>
      <FIELD name="Transmission" utype="spec:Data.FluxAxis.Value" ucd="phys.transmission" unit="" datatype="float"/>
      <DATA>
        <TABLEDATA>
          <TR>
            <TD>19420.0</TD>
            <TD>0.0000000000</TD>
          </TR>
          <TR>
            <TD>19570.0</TD>
            <TD>0.0280000000</TD>
          </TR>
          <TR>
            <TD>19710.0</TD>
            <TD>0.1210000000</TD>
          </TR>
          <TR>
            <TD>19750.0</TD>
            <TD>0.2420000000</TD>
          </TR>
          <TR>
            <TD>19810.0</TD>
            <TD>0.3620000000</TD>
          </TR>
          <TR>
            <TD>19860.0</TD>
            <TD>0.4830000000</TD>
          </TR>
          <TR>
            <TD>19870.0</TD>
            <TD>0.5720000000</TD>
          </TR>
          <TR>
            <TD>19950.0</TD>
            <TD>0.5920000000</TD>
          </TR>
          <TR>
            <TD>19990.0</TD>
            <TD>0.6440000000</TD>
          </TR>
          <TR>
            <TD>20060.0</TD>
            <TD>0.3220000000</TD>
          </TR>
          <TR>
            <TD>20120.0</TD>
            <TD>0.4670000000</TD>
          </TR>
          <TR>
            <TD>20150.0</TD>
            <TD>0.4310000000</TD>
          </TR>
          <TR>
            <TD>20210.0</TD>
            <TD>0.5640000000</TD>
          </TR>
          <TR>
            <TD>20270.0</TD>
            <TD>0.7250000000</TD>
          </TR>
          <TR>
            <TD>20360.0</TD>
            <TD>0.8700000000</TD>
          </TR>
          <TR>
            <TD>20630.0</TD>
            <TD>0.7490000000</TD>
          </TR>
          <TR>
            <TD>20870.0</TD>
            <TD>0.8860000000</TD>
          </TR>
          <TR>
            <TD>21080.0</TD>
            <TD>0.9340000000</TD>
          </TR>
          <TR>
            <TD>21230.0</TD>
            <TD>0.9340000000</TD>
          </TR>
          <TR>
            <TD>21390.0</TD>
            <TD>0.9660000000</TD>
          </TR>
          <TR>
            <TD>21540.0</TD>
            <TD>0.9460000000</TD>
          </TR>
          <TR>
            <TD>21950.0</TD>
            <TD>0.9780000000</TD>
          </TR>
          <TR>
            <TD>22010.0</TD>
            <TD>0.9640000000</TD>
          </TR>
          <TR>
            <TD>22190.0</TD>
            <TD>1.0000000000</TD>
          </TR>
          <TR>
            <TD>22250.0</TD>
            <TD>1.0000000000</TD>
          </TR>
          <TR>
            <TD>22670.0</TD>
            <TD>0.9600000000</TD>
          </TR>
          <TR>
            <TD>22780.0</TD>
            <TD>0.9700000000</TD>
          </TR>
          <TR>
            <TD>22970.0</TD>
            <TD>0.8860000000</TD>
          </TR>
          <TR>
            <TD>23030.0</TD>
            <TD>0.7250000000</TD>
          </TR>
          <TR>
            <TD>23140.0</TD>
            <TD>0.5640000000</TD>
          </TR>
          <TR>
            <TD>23200.0</TD>
            <TD>0.4030000000</TD>
          </TR>
          <TR>
            <TD>23290.0</TD>
            <TD>0.2420000000</TD>
          </TR>
          <TR>
            <TD>23500.0</TD>
            <TD>0.0810000000</TD>
          </TR>
          <TR>
            <TD>23810.0</TD>
            <TD>0.0160000000</TD>
          </TR>
          <TR>
            <TD>24260.0</TD>
            <TD>0.0000000000</TD>
          </TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
