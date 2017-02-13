<?xml version="1.0"?>
<VOTABLE version="1.1" xsi:schemaLocation="http://www.ivoa.net/xml/VOTable/v1.1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <INFO name="QUERY_STATUS" value="OK"/>
  <RESOURCE type="results">
    <TABLE utype="photdm:PhotometryFilter.transmissionCurve.spectrum">
    <PARAM name="FilterProfileService" value="ivo://svo/fps" ucd="meta.ref.ivorn" utype="PhotometryFilter.fpsIdentifier" datatype="char" arraysize="*"/>
    <PARAM name="filterID" value="DENIS/DENIS.I" ucd="meta.id" utype="photdm:PhotometryFilter.identifier" datatype="char" arraysize="*"/>
    <PARAM name="WavelengthUnit" value="Angstrom" ucd="meta.unit" utype="PhotometryFilter.SpectralAxis.unit" datatype="char" arraysize="*"/>
    <PARAM name="WavelengthUCD" value="em.wl" ucd="meta.ucd" utype="PhotometryFilter.SpectralAxis.UCD" datatype="char" arraysize="*"/>
    <PARAM name="Description" value="DENIS I" ucd="meta.note" utype="photdm:PhotometryFilter.description" datatype="char" arraysize="*"/>
    <PARAM name="PhotSystem" value="DENIS" utype="photdm:PhotometricSystem.description" datatype="char" arraysize="*">
       <DESCRIPTION>Photometric system</DESCRIPTION>
    </PARAM>
    <PARAM name="Band" value="I" utype="photdm:PhotometryFilter.bandName" datatype="char" arraysize="*"/>
    <PARAM name="ProfileReference" value="http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_148/index_148.html" datatype="char" arraysize="*"/>
    <PARAM name="Description" value="DENIS I" ucd="meta.note" utype="photdm:PhotometryFilter.description" datatype="char" arraysize="*"/>
    <PARAM name="WavelengthMean" value="7930.36671154" unit="Angstrom" ucd="em.wl" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Location.Value" datatype="float" >
       <DESCRIPTION>Mean wavelength. Defined as integ[x*filter(x) dx]/integ[filter(x) dx]</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthEff" value="7862.1015966" unit="Angstrom" ucd="em.wl.effective" datatype="float" >
       <DESCRIPTION>Effective wavelength. Defined as integ[x*filter(x)*vega(x) dx]/integ[filter(x)*vega(x) dx]</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthMin" value="7007.28395062" unit="Angstrom" ucd="em.wl;stat.min" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Start" datatype="float" >
       <DESCRIPTION>Minimum filter wavelength. Defined as the first lambda value with a transmission at least 1% of maximum transmission</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthMax" value="9140" unit="Angstrom" ucd="em.wl;stat.max" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Stop" datatype="float" >
       <DESCRIPTION>Maximum filter wavelength. Defined as the last lambda value with a transmission at least 1% of maximum transmission</DESCRIPTION>
    </PARAM>
    <PARAM name="WidthEff" value="1132.54838441" unit="Angstrom" ucd="instr.bandwidth" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Extent" datatype="float" >
       <DESCRIPTION>Effective width. Defined as integ[x*filter(x) dx].\nEquivalent to the horizontal size of a rectangle with height equal to maximum transmission and with the same area that the one covered by the filter transmission curve.</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthCen" value="7943.95061728" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Central wavelength. Defined as the central wavelength between the two points defining FWMH</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthPivot" value="7919.24477085" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Peak wavelength. Defined as sqrt{integ[x*filter(x) dx]/integ[filter(x) dx/x]}</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthPeak" value="7690" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Peak wavelength. Defined as the lambda value with larger transmission</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthPhot" value="7883.85237653" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Photon distribution based effective wavelength. Defined as integ[x^2*filter(x)*vega(x) dx]/integ[x*filter(x)*vega(x) dx]</DESCRIPTION>
    </PARAM>
    <PARAM name="FWHM" value="1311.11111111" unit="Angstrom" ucd="instr.bandwidth" datatype="float" >
       <DESCRIPTION>Full width at half maximum. Defined as the difference between the two wavelengths for which filter transmission is half maximum</DESCRIPTION>
    </PARAM>
    <PARAM name="PhotCalID" value="DENIS/DENIS.I/Vega" ucd="meta.id" utype="photdm:PhotCal.identifier" datatype="char" arraysize="*"/>
    <PARAM name="MagSys" value="Vega" ucd="meta.code" utype="photdm:PhotCal.MagnitudeSystem.type" datatype="char" arraysize="*"/>
    <PARAM name="ZeroPoint" value="2442.22329131" unit="Jy" ucd="phot.flux.density" utype="photdm:PhotCal.ZeroPoint.Flux.value" datatype="float" />
    <PARAM name="ZeroPointUnit" value="Jy" ucd="meta.unit" utype="photdm:PhotCal.ZeroPoint.Flux.unit" datatype="char" arraysize="*"/>
    <PARAM name="ZeroPointType" value="Pogson" ucd="meta.code" utype="photdm:PhotCal.ZeroPoint.type" datatype="char" arraysize="*"/>
      <FIELD name="Wavelength" utype="spec:Data.SpectralAxis.Value" ucd="em.wl" unit="Angstrom" datatype="float"/>
      <FIELD name="Transmission" utype="spec:Data.FluxAxis.Value" ucd="phys.transmission" unit="" datatype="float"/>
      <DATA>
        <TABLEDATA>
          <TR>
            <TD>6990.0</TD>
            <TD>0.0000000000</TD>
          </TR>
          <TR>
            <TD>7130.0</TD>
            <TD>0.0810000000</TD>
          </TR>
          <TR>
            <TD>7170.0</TD>
            <TD>0.1610000000</TD>
          </TR>
          <TR>
            <TD>7190.0</TD>
            <TD>0.2420000000</TD>
          </TR>
          <TR>
            <TD>7220.0</TD>
            <TD>0.3220000000</TD>
          </TR>
          <TR>
            <TD>7250.0</TD>
            <TD>0.4030000000</TD>
          </TR>
          <TR>
            <TD>7280.0</TD>
            <TD>0.4830000000</TD>
          </TR>
          <TR>
            <TD>7320.0</TD>
            <TD>0.5640000000</TD>
          </TR>
          <TR>
            <TD>7340.0</TD>
            <TD>0.6440000000</TD>
          </TR>
          <TR>
            <TD>7350.0</TD>
            <TD>0.7250000000</TD>
          </TR>
          <TR>
            <TD>7400.0</TD>
            <TD>0.8050000000</TD>
          </TR>
          <TR>
            <TD>7430.0</TD>
            <TD>0.8860000000</TD>
          </TR>
          <TR>
            <TD>7540.0</TD>
            <TD>0.9970000000</TD>
          </TR>
          <TR>
            <TD>7600.0</TD>
            <TD>0.3780000000</TD>
          </TR>
          <TR>
            <TD>7690.0</TD>
            <TD>1.0000000000</TD>
          </TR>
          <TR>
            <TD>7850.0</TD>
            <TD>0.9660000000</TD>
          </TR>
          <TR>
            <TD>8000.0</TD>
            <TD>0.8860000000</TD>
          </TR>
          <TR>
            <TD>8080.0</TD>
            <TD>0.8050000000</TD>
          </TR>
          <TR>
            <TD>8160.0</TD>
            <TD>0.7250000000</TD>
          </TR>
          <TR>
            <TD>8220.0</TD>
            <TD>0.6600000000</TD>
          </TR>
          <TR>
            <TD>8430.0</TD>
            <TD>0.6600000000</TD>
          </TR>
          <TR>
            <TD>8560.0</TD>
            <TD>0.5640000000</TD>
          </TR>
          <TR>
            <TD>8610.0</TD>
            <TD>0.4830000000</TD>
          </TR>
          <TR>
            <TD>8640.0</TD>
            <TD>0.4030000000</TD>
          </TR>
          <TR>
            <TD>8670.0</TD>
            <TD>0.3220000000</TD>
          </TR>
          <TR>
            <TD>8700.0</TD>
            <TD>0.2420000000</TD>
          </TR>
          <TR>
            <TD>8750.0</TD>
            <TD>0.1610000000</TD>
          </TR>
          <TR>
            <TD>8840.0</TD>
            <TD>0.0810000000</TD>
          </TR>
          <TR>
            <TD>8990.0</TD>
            <TD>0.0200000000</TD>
          </TR>
          <TR>
            <TD>9290.0</TD>
            <TD>0.0000000000</TD>
          </TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
