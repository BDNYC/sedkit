<?xml version="1.0"?>
<VOTABLE version="1.1" xsi:schemaLocation="http://www.ivoa.net/xml/VOTable/v1.1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <INFO name="QUERY_STATUS" value="OK"/>
  <RESOURCE type="results">
    <TABLE utype="photdm:PhotometryFilter.transmissionCurve.spectrum">
    <PARAM name="FilterProfileService" value="ivo://svo/fps" ucd="meta.ref.ivorn" utype="PhotometryFilter.fpsIdentifier" datatype="char" arraysize="*"/>
    <PARAM name="filterID" value="DENIS/DENIS.J" ucd="meta.id" utype="photdm:PhotometryFilter.identifier" datatype="char" arraysize="*"/>
    <PARAM name="WavelengthUnit" value="Angstrom" ucd="meta.unit" utype="PhotometryFilter.SpectralAxis.unit" datatype="char" arraysize="*"/>
    <PARAM name="WavelengthUCD" value="em.wl" ucd="meta.ucd" utype="PhotometryFilter.SpectralAxis.UCD" datatype="char" arraysize="*"/>
    <PARAM name="Description" value="DENIS J" ucd="meta.note" utype="photdm:PhotometryFilter.description" datatype="char" arraysize="*"/>
    <PARAM name="PhotSystem" value="DENIS" utype="photdm:PhotometricSystem.description" datatype="char" arraysize="*">
       <DESCRIPTION>Photometric system</DESCRIPTION>
    </PARAM>
    <PARAM name="Band" value="J" utype="photdm:PhotometryFilter.bandName" datatype="char" arraysize="*"/>
    <PARAM name="ProfileReference" value="http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_148/index_148.html" datatype="char" arraysize="*"/>
    <PARAM name="Description" value="DENIS J" ucd="meta.note" utype="photdm:PhotometryFilter.description" datatype="char" arraysize="*"/>
    <PARAM name="WavelengthMean" value="12357.4447458" unit="Angstrom" ucd="em.wl" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Location.Value" datatype="float" >
       <DESCRIPTION>Mean wavelength. Defined as integ[x*filter(x) dx]/integ[filter(x) dx]</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthEff" value="12210.6027581" unit="Angstrom" ucd="em.wl.effective" datatype="float" >
       <DESCRIPTION>Effective wavelength. Defined as integ[x*filter(x)*vega(x) dx]/integ[filter(x)*vega(x) dx]</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthMin" value="10508.3333333" unit="Angstrom" ucd="em.wl;stat.min" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Start" datatype="float" >
       <DESCRIPTION>Minimum filter wavelength. Defined as the first lambda value with a transmission at least 1% of maximum transmission</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthMax" value="13980" unit="Angstrom" ucd="em.wl;stat.max" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Stop" datatype="float" >
       <DESCRIPTION>Maximum filter wavelength. Defined as the last lambda value with a transmission at least 1% of maximum transmission</DESCRIPTION>
    </PARAM>
    <PARAM name="WidthEff" value="2040.72095279" unit="Angstrom" ucd="instr.bandwidth" utype="photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Extent" datatype="float" >
       <DESCRIPTION>Effective width. Defined as integ[x*filter(x) dx].\nEquivalent to the horizontal size of a rectangle with height equal to maximum transmission and with the same area that the one covered by the filter transmission curve.</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthCen" value="12492.345679" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Central wavelength. Defined as the central wavelength between the two points defining FWMH</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthPivot" value="12336.75467" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Peak wavelength. Defined as sqrt{integ[x*filter(x) dx]/integ[filter(x) dx/x]}</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthPeak" value="12770" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Peak wavelength. Defined as the lambda value with larger transmission</DESCRIPTION>
    </PARAM>
    <PARAM name="WavelengthPhot" value="12253.0619335" unit="Angstrom" ucd="em.wl" datatype="float" >
       <DESCRIPTION>Photon distribution based effective wavelength. Defined as integ[x^2*filter(x)*vega(x) dx]/integ[x*filter(x)*vega(x) dx]</DESCRIPTION>
    </PARAM>
    <PARAM name="FWHM" value="1991.11111111" unit="Angstrom" ucd="instr.bandwidth" datatype="float" >
       <DESCRIPTION>Full width at half maximum. Defined as the difference between the two wavelengths for which filter transmission is half maximum</DESCRIPTION>
    </PARAM>
    <PARAM name="PhotCalID" value="DENIS/DENIS.J/Vega" ucd="meta.id" utype="photdm:PhotCal.identifier" datatype="char" arraysize="*"/>
    <PARAM name="MagSys" value="Vega" ucd="meta.code" utype="photdm:PhotCal.MagnitudeSystem.type" datatype="char" arraysize="*"/>
    <PARAM name="ZeroPoint" value="1588.32895725" unit="Jy" ucd="phot.flux.density" utype="photdm:PhotCal.ZeroPoint.Flux.value" datatype="float" />
    <PARAM name="ZeroPointUnit" value="Jy" ucd="meta.unit" utype="photdm:PhotCal.ZeroPoint.Flux.unit" datatype="char" arraysize="*"/>
    <PARAM name="ZeroPointType" value="Pogson" ucd="meta.code" utype="photdm:PhotCal.ZeroPoint.type" datatype="char" arraysize="*"/>
      <FIELD name="Wavelength" utype="spec:Data.SpectralAxis.Value" ucd="em.wl" unit="Angstrom" datatype="float"/>
      <FIELD name="Transmission" utype="spec:Data.FluxAxis.Value" ucd="phys.transmission" unit="" datatype="float"/>
      <DATA>
        <TABLEDATA>
          <TR>
            <TD>10200.0</TD>
            <TD>0.0000000000</TD>
          </TR>
          <TR>
            <TD>10570.0</TD>
            <TD>0.0120000000</TD>
          </TR>
          <TR>
            <TD>10740.0</TD>
            <TD>0.0810000000</TD>
          </TR>
          <TR>
            <TD>10770.0</TD>
            <TD>0.1610000000</TD>
          </TR>
          <TR>
            <TD>10800.0</TD>
            <TD>0.2420000000</TD>
          </TR>
          <TR>
            <TD>10830.0</TD>
            <TD>0.2660000000</TD>
          </TR>
          <TR>
            <TD>11000.0</TD>
            <TD>0.2980000000</TD>
          </TR>
          <TR>
            <TD>11100.0</TD>
            <TD>0.4030000000</TD>
          </TR>
          <TR>
            <TD>11210.0</TD>
            <TD>0.2580000000</TD>
          </TR>
          <TR>
            <TD>11390.0</TD>
            <TD>0.4030000000</TD>
          </TR>
          <TR>
            <TD>11480.0</TD>
            <TD>0.4830000000</TD>
          </TR>
          <TR>
            <TD>11560.0</TD>
            <TD>0.5640000000</TD>
          </TR>
          <TR>
            <TD>11570.0</TD>
            <TD>0.6440000000</TD>
          </TR>
          <TR>
            <TD>11590.0</TD>
            <TD>0.7250000000</TD>
          </TR>
          <TR>
            <TD>11680.0</TD>
            <TD>0.8210000000</TD>
          </TR>
          <TR>
            <TD>11740.0</TD>
            <TD>0.8170000000</TD>
          </TR>
          <TR>
            <TD>11940.0</TD>
            <TD>0.9100000000</TD>
          </TR>
          <TR>
            <TD>12040.0</TD>
            <TD>0.8370000000</TD>
          </TR>
          <TR>
            <TD>12190.0</TD>
            <TD>0.9760000000</TD>
          </TR>
          <TR>
            <TD>12420.0</TD>
            <TD>0.9140000000</TD>
          </TR>
          <TR>
            <TD>12770.0</TD>
            <TD>1.0000000000</TD>
          </TR>
          <TR>
            <TD>13100.0</TD>
            <TD>0.9020000000</TD>
          </TR>
          <TR>
            <TD>13220.0</TD>
            <TD>0.9020000000</TD>
          </TR>
          <TR>
            <TD>13340.0</TD>
            <TD>0.8050000000</TD>
          </TR>
          <TR>
            <TD>13400.0</TD>
            <TD>0.7250000000</TD>
          </TR>
          <TR>
            <TD>13450.0</TD>
            <TD>0.6440000000</TD>
          </TR>
          <TR>
            <TD>13480.0</TD>
            <TD>0.5640000000</TD>
          </TR>
          <TR>
            <TD>13490.0</TD>
            <TD>0.4830000000</TD>
          </TR>
          <TR>
            <TD>13510.0</TD>
            <TD>0.4030000000</TD>
          </TR>
          <TR>
            <TD>13520.0</TD>
            <TD>0.3220000000</TD>
          </TR>
          <TR>
            <TD>13540.0</TD>
            <TD>0.2420000000</TD>
          </TR>
          <TR>
            <TD>13550.0</TD>
            <TD>0.1610000000</TD>
          </TR>
          <TR>
            <TD>13570.0</TD>
            <TD>0.0810000000</TD>
          </TR>
          <TR>
            <TD>13670.0</TD>
            <TD>0.0120000000</TD>
          </TR>
          <TR>
            <TD>13800.0</TD>
            <TD>0.0560000000</TD>
          </TR>
          <TR>
            <TD>13860.0</TD>
            <TD>0.0120000000</TD>
          </TR>
          <TR>
            <TD>14580.0</TD>
            <TD>0.0000000000</TD>
          </TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
