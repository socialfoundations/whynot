// This code was taken from: http://bit-player.org/extras/limits/ltg.html
// and modified to remove (comment out) the graphics rendering.

/*  Limits to Growth: This is a re-implementation in JavaScript
    of World3, the social-economic-environmental model created by
    Dennis and Donella Meadows and others circa 1970. The results
    of the modeling exercise were published in The Limits to Growth
    in 1972, and the model itself was more fully documented in
    Dynamics of Growth in a Finite World in 1974. 
    
*/



// The DYNAMO clip function, a poor-man's
// conditional expression.

function clip(a, b, x, y) {
	if (x >= y) 
		return a;
	else
		return b
}

// when we create an Equation with qNumber n, it becomes element qArray[n]
// note that there is no qArray[0]

var qArray = new Array();


// Equations with qClass "Level" are pushed onto this Array

var levelArray = new Array();


// Equations with qClass "Rate" are pushed onto this Array

var rateArray = new Array();


// Equations with qClass "Aux" are pushed onto this Array

var auxArray = new Array();


// construtor for Level objects

var Level = function(qName, qNumber, initVal) {
  this.qName = qName;
  this.qNumber = qNumber;
  this.qType = "Level";
  this.units = "dimensionless";
  this.initVal = initVal || null;
  this.j = this.k = this.initVal;
  this.plotThisVar = false;
  this.plotColor = "transparent";  // default to be overridden
  this.plotMin = 0;         // default to be overridden
  this.plotMax = 1000;      // default to be overridden
  this.data = [ {x: startTime, y: this.k} ];
  qArray[qNumber] = this;
  levelArray.push(this);
}

Level.prototype.reset = function() {
  this.j = this.k = this.initVal;
  this.data = [ {x: startTime, y: this.k} ];
}

Level.prototype.warmup = function() {
  this.k = this.updateFn();
}

Level.prototype.update = function() {
  this.k = this.updateFn();
  this.data.push( {x: t, y: this.k} );
  if (this.plotThisVar) {
    this.plot();
  }
  return this.k;
}

Level.prototype.tick = function() {
  this.j = this.k;
}

Level.prototype.plot = function() {
//  var cvx = document.getElementById("cv").getContext("2d");
//  cvx.strokeStyle = this.plotColor;
//  cvx.lineWidth = 2;
//  cvx.beginPath();
//  var leftPoint = this.data[0];
//  cvx.moveTo(scaleX(leftPoint.x, startTime, stopTime), scaleY(leftPoint.y, this.plotMin, this.plotMax));
//  for (var i = 1 ; i < this.data.length ; i++) {
//    var p = this.data[i];
//    cvx.lineTo(scaleX(p.x, startTime, stopTime), scaleY(p.y, this.plotMin, this.plotMax));
//  }
//  cvx.stroke();
//  cvx.closePath();
}


// construtor for Rate objects

var Rate = function(qName, qNumber) {
  this.qName = qName;
  this.qNumber = qNumber;
  this.qType = "Rate";
  this.units = "dimensionless";
  this.j = this.k = null;
  this.plotThisVar = false;
  this.plotColor = "transparent";  // default to be overridden
  this.plotMin = 0;         // default to be overridden
  this.plotMax = 1000;      // default to be overridden
  this.data = [];
  qArray[qNumber] = this;
  rateArray.push(this);
}

Rate.prototype.reset = function() {
  this.j = this.k = null;
  this.data = [];
}

Rate.prototype.warmup = Level.prototype.warmup;

Rate.prototype.update = Level.prototype.update;

Rate.prototype.tick = Level.prototype.tick;

Rate.prototype.plot = Level.prototype.plot;




// constructor for Aux objects

var Aux = function(qName, qNumber) {
  this.qName = qName;
  this.qNumber = qNumber;
  this.qType = "Aux";
  this.units = "dimensionless";
  this.dependencies = [];
  this.j = this.k = null;
  this.plotColor = "transparent";  // default to be overridden
  this.plotMin = 0;         // default to be overridden
  this.plotMax = 1000;      // default to be overridden
  this.data = [];
  qArray[qNumber] = this;
  auxArray.push(this);
}
  
Aux.prototype.reset = function() {
  this.j = this.k = null;
  this.data = [];
}

Aux.prototype.warmup = Level.prototype.warmup;

Aux.prototype.update = Level.prototype.update;

Aux.prototype.tick = Level.prototype.tick;

Aux.prototype.plot = Level.prototype.plot;




// constructor for Smooth objects

var Smooth = function(qName, qNumber, delay) {
  this.qName = qName;
  this.qNumber = qNumber;
  this.qType = "Aux";
  this.units = "dimensionless";
  this.dependencies = [];
  this.del = delay;
  this.firstCall = true;
  this.j = this.k = null;
  qArray[qNumber] = this;
  auxArray.push(this);
}

Smooth.prototype.init = function() {
  this.theInput = this.initFn();
  this.j = this.k = this.theInput.k || this.initVal; 
}
  
Smooth.prototype.reset = function() {
  this.firstCall = true;
  this.j = this.k = this.null;
}

Smooth.prototype.update = function() {
  if (this.firstCall) {
    this.j = this.k = this.theInput.k || this.initVal;
    this.firstCall = false;
    return this.k;
  }
  else {
    this.k = this.j + dt * (this.theInput.j - this.j) / this.del;
    return this.k;
  }
}

Smooth.prototype.warmup = Smooth.prototype.init;

Smooth.prototype.tick = Level.prototype.tick;




// constructor for Delay3 objects
// third-order exponential delay for Rate variables


var Delay3 = function(qName, qNumber, delay) {
  this.qName = qName;
  this.qNumber = qNumber;
  this.qType = "Aux";
  this.units = "dimensionless";
  this.dependencies = [];
  this.delayPerStage = delay / 3;
  this.firstCall = true;
  this.j = this.k = null;
  this.alpha = { j: null, k: null };
  this.beta  = { j: null, k: null };
  this.gamma = { j: null, k: null };
  qArray[qNumber] = this;
  auxArray.push(this);
}

Delay3.prototype.init = function() {
  this.theInput = this.initFn();
  this.j = this.k = this.theInput.k;
  this.alpha.j = this.alpha.k = this.theInput.j;
  this.beta.j  = this.beta.k  = this.theInput.j;
  this.gamma.j = this.gamma.k = this.theInput.j;
}


Delay3.prototype.reset = function() {
  this.firstCall = true;
  this.j = this.k = null;
  this.alpha = { j: null, k: null };
  this.beta  = { j: null, k: null };
  this.gamma = { j: null, k: null };
}

Delay3.prototype.update = function() {
  if (this.firstCall) {
    this.j = this.k = this.theInput.k;
    this.alpha.j = this.alpha.k = this.theInput.k;
    this.beta.j  = this.beta.k  = this.theInput.k;
    this.gamma.j = this.gamma.k = this.theInput.k;
    this.firstCall = false;
    return this.k;    
  }
  else {
    this.alpha.k = this.alpha.j + dt * (this.theInput.j - this.alpha.j) / this.delayPerStage;
    this.beta.k  = this.beta.j  + dt * (this.alpha.j    - this.beta.j)  / this.delayPerStage;
    this.gamma.k = this.gamma.j + dt * (this.beta.j     - this.gamma.j) / this.delayPerStage;
    this.alpha.j = this.alpha.k
    this.beta.j  = this.beta.k
    this.gamma.j = this.gamma.k
    this.k = this.gamma.k
    return this.k;
  }
}

Delay3.prototype.warmup = Delay3.prototype.init;

Delay3.prototype.tick = Level.prototype.tick;


// constructor for Table objects

var Table = function(qName, qNumber, data, iMin, iMax, iDelta) {
  this.qName = qName;
  this.qNumber = qNumber;
  this.qType = "Aux";
  this.units = "dimensionless";
  this.dependencies = [];
  this.data = data;
  this.iMin = iMin;
  this.iMax = iMax;
  this.iDelta = iDelta;
  this.indices = [];
  for (var i = iMin ; i <= iMax ; i += iDelta)
    this.indices.push(i);
  this.k = this.j = null;
  qArray[qNumber] = this;
  auxArray.push(this);
}

Table.prototype.interpolate = function(lower, upper, fraction) {
  var lowerVal = this.data[lower];
  var upperVal = this.data[upper];
  return lowerVal + (fraction * (upperVal - lowerVal));
}

Table.prototype.lookup = function(v) {
  if (v <= this.iMin) {
    return this.data[0];
  }
  else if (v >= this.iMax) {
    return this.data[this.data.length - 1];
  }
  else {
    for (var i = this.iMin, j = 0 ; i <= this.iMax ; i += this.iDelta, j++)
      if (i >= v) {
        return this.interpolate(j-1, j, (v - (i - this.iDelta)) / this.iDelta);
      }
  }
}

Table.prototype.reset = function() { return null; }

Table.prototype.update = function() {
  this.k = this.lookup(this.updateFn());
  return this.k; 
}

Table.prototype.warmup = Table.prototype.update;

Table.prototype.tick = Level.prototype.tick;



// sort the Aux equations into an order such that each one will
// not be executed until all of its dependencies have been satisfied

var gatherDependencies = function() {
  var depArray = [];
  for (var i = 0 ; i < auxArray.length ; i++) {
    var d = new Object();
    d[auxArray[i].qName] = auxArray[i].dependencies;
    depArray[i] = d;
    }
  return depArray;  
}

var printDeps = function() {
  for (var i = 0 ; i < auxArray.length ; i++) {
    document.writeln(auxArray[i].qName + "<br/>");
    for (var j = 0 ; j < auxArray[i].dependencies.length ; j++) {
      document.writeln("____" + auxArray[i].dependencies[j] + "<br/>");
    }
  }    
}

var qNameToQNumber = function(theName) {
  for (var i = 1 ; i < qArray.length ; i++) {
    if (qArray[i].qName === theName)
      return qArray[i].qNumber;
  }
}

var printQNumberDependencies = function() {
  for (var i = 0 ; i < auxArray.length ; i++) {
    document.write(auxArray[i].qNumber, " ");
    for (var j = 0 ; j < auxArray[i].dependencies.length ; j++) {
      var qN = qNameToQNumber(auxArray[i].dependencies[j]);
//      console.log(i, j, auxArray[i].qNumber, qN);
      document.write(qN, " ");
    }
    document.write("</br>");
  }    
}


var printQNameDependencies = function() {
  for (var i = 0 ; i < auxArray.length ; i++) {
    document.write(auxArray[i].qName, " ");
    for (var j = 0 ; j < auxArray[i].dependencies.length ; j++) {
      document.write(auxArray[i].dependencies[j], " ");
    }
    document.write("</br>");
  }    
}




var auxSequence = ["population", "deathsPerYear", "lifetimeMultiplierFromCrowding", "industrialCapitalOutputRatio", "averageLifetimeOfIndustrialCapital", "averageLifetimeOfServiceCapital", "serviceCapitalOutputRatio", "laborForce", "landFractionCultivated", "developmentCostPerHectare", "landYieldFactor", "nonrenewableResourceUsageFactor", "nonrenewableResourceFractionRemaining", "persistentPollutionGenerationFactor", "indexOfPersistentPollution", "fractionOfIndustrialOutputAllocatedToConsumptionConstant", "averageLifetimeOfAgriculturalInputs", "laborUtilizationFractionDelayed", "agriculturalInputs", "perceivedFoodRatio", "fractionOfPopulationUrban", "crudeDeathRate", "crudeBirthRate", "fractionOfCapitalAllocatedToObtainingResourcesBefore", "fractionOfCapitalAllocatedToObtainingResourcesAfter", "fractionOfCapitalAllocatedToObtainingResources", "lifetimeMultiplierFromPollution", "landFertilityDegradationRate", "capitalUtilizationFraction", "industrialOutput", "industrialOutputPerCapita", "delayedIndustrialOutputPerCapita", "socialFamilySizeNorm", "averageIndustrialOutputPerCapita", "familyIncomeExpectation", "familyResponseToSocialNorm", "desiredCompletedFamilySize", "crowdingMultiplierFromIndustrialization", "indicatedServiceOutputPerCapitaBefore", "indicatedServiceOutputPerCapitaAfter", "indicatedServiceOutputPerCapita", "fractionOfIndustrialOutputAllocatedToConsumptionVariable", "fractionOfIndustrialOutputAllocatedToConsumption", "jobsPerIndustrialCapitalUnit", "potentialJobsInIndustrialSector", "serviceOutput", "serviceOutputPerCapita", "fractionOfIndustrialOutputAllocatedToServicesBefore", "fractionOfIndustrialOutputAllocatedToServicesAfter", "fractionOfIndustrialOutputAllocatedToServices", "jobsPerServiceCapitalUnit", "potentialJobsInServiceSector", "healthServicesAllocationsPerCapita", "effectiveHealthServicesPerCapita", "lifetimeMultiplierFromHealthServicesBefore", "lifetimeMultiplierFromHealthServicesAfter", "lifetimeMultiplierFromHealthServices", "fractionOfInputsAllocatedToLandMaintenance", "agriculturalInputsPerHectare", "jobsPerHectare", "potentialJobsInAgriculturalSector", "jobs", "laborUtilizationFraction", "landYieldMultiplierFromCapital", "landYieldMultiplierFromAirPollutionBefore", "landYieldMultiplierFromAirPollutionAfter", "landYieldMultiplierFromAirPollution", "landYield", "marginalProductivityOfLandDevelopment", "marginalLandYieldMultiplierFromCapital", "marginalProductivityOfAgriculturalInputs", "fractionOfInputsAllocatedToLandDevelopment", "food", "foodPerCapita", "indicatedFoodPerCapitaBefore", "indicatedFoodPerCapitaAfter", "indicatedFoodPerCapita", "fractionOfIndustrialOutputAllocatedToAgricultureBefore", "fractionOfIndustrialOutputAllocatedToAgricultureAfter", "fractionOfIndustrialOutputAllocatedToAgriculture", "totalAgriculturalInvestment", "currentAgriculturalInputs", "foodRatio", "landFertilityRegenerationTime", "lifetimeMultiplierFromFood", "lifeExpectancy", "mortality0To14", "mortality15To44", "mortality45To64", "mortality65AndOver", "fecundityMultiplier", "perceivedLifeExpectancy", "compensatoryMultiplierFromPerceivedLifeExpectancy", "maxTotalFertility", "desiredTotalFertility", "needForFertilityControl", "fractionOfServicesAllocatedToFertilityControl", "fertilityControlAllocationPerCapita", "fertilityControlFacilitiesPerCapita", "fertilityControlEffectiveness", "totalFertility", "landLifeMultiplierFromYieldBefore", "landLifeMultiplierFromYieldAfter", "landLifeMultiplierFromYield", "averageLifeOfLand", "urbanIndustrialLandPerCapita", "urbanIndustrialLandRequired", "perCapitaResourceUsageMultiplier", "persistentPollutionGeneratedByIndustrialOutput", "persistentPollutionGeneratedByAgriculturalOutput", "assimilationHalfLifeMultiplier", "assimilationHalfLife", "fractionOfIndustrialOutputAllocatedToIndustry", "fractionOfOutputInAgriculture", "fractionOfOutputInIndustry", "fractionOfOutputInServices"]


var sortAuxEqns = function() {
  for (var i = 0 ; i < auxSequence.length ; i++) {
    eval(auxSequence[i]).sequenceNumber = i;
  }
  auxArray.sort(function(left, right) {
                  if (left.sequenceNumber < right.sequenceNumber) { return -1; } else { return 1; }
                })  
}




// PARAMETERS THAT GOVERN THE RUNNING OF THE MODEL

var startTime = 1900;
var stopTime = 2100;
var t = 1900;
var dt = 1.0;
var policyYear = 1975;                 // eqn 150.1
var plotInterval = Math.max(dt, 1);

var resetModel = function() {
  t = startTime;
  for (var i = 1 ; i < qArray.length ; i++) {
    qArray[i].reset();
  }
  setUpGraph();
}


var initSmoothsAndDelay3s = function() {
  for (var i = 1 ; i < qArray.length ; i++) {
    var q = qArray[i];
    if (q.constructor === Smooth || q.constructor === Delay3) {
      q.init();
    }
  }
}


var updateAuxen = function() {
  for (var i = 0 ; i < auxArray.length ; i++) {
    auxArray[i].update();
//    console.log(i);
  }
}


var updateRates = function() {
  for (var i = 0 ; i < rateArray.length ; i++) {
    rateArray[i].update();
  }
}


var updateLevels = function() {
  for (var i = 0 ; i < levelArray.length ; i++) {
    levelArray[i].update();
  }
}

var warmupAuxen = function() {
  for (var i = 0 ; i < auxArray.length ; i++) {
    auxArray[i].warmup();
//    console.log(auxArray[i].qName, auxArray[i].k);
  }
}


var warmupRates = function() {
  for (var i = 0 ; i < rateArray.length ; i++) {
    rateArray[i].warmup();
  }
}


var warmupLevels = function() {
  for (var i = 0 ; i < levelArray.length ; i++) {
    levelArray[i].warmup();
  }
}

var tock = function() {
  for (var i = 1 ; i < qArray.length ; i++) {
    qArray[i].tick();
  }
}








var initModel = function() {
  initSmoothsAndDelay3s();
  sortAuxEqns();
  t = startTime;
}

var timeStep = function() {
  t += dt;
  updateLevels();
  updateAuxen();
  updateRates();
  tock();
}



var animationStep = function() {
  timeStep();
  if (t > stopTime) {
    clearInterval(plotTimer);
    enableControls();
    setRunButton();
  }
}

var stopModel = function() {
  clearInterval(plotTimer);
  enableControls();
  setRunButton();
}




var runModel = function() {
  var plotDelay = 0 * dt;  // milliseconds
  disableControls();
  setStopButton();
  resetModel();
  initModel();
  setUpGraph();
  for (var i = 1 ; i <= 3 ; i++) {
    warmupAuxen();
    warmupRates();
    tock();
  }
  for (var i = 1 ; i <= 10 ; i++) {
    warmupAuxen();
    warmupRates();
    warmupLevels();
    tock();
  }
  for (var i = 0 ; i < levelArray.length ; i++) {
    levelArray[i].reset();
  }
  plotTimer = setInterval(animationStep, plotDelay);   // note GLOBAL
}

var fastRun = function() {
  resetModel();
  initModel();
  setUpGraph();
  for (var i = 1 ; i <= 100 ; i++) {
    warmupAuxen();
    warmupRates();
    tock();
  }
  while (t < stopTime) {
    timeStep();
  }  
}


var checkForNaNs = function() {
  for (var i=1 ; i < qArray.length ; i++) { if (isNaN(qArray[i].k)) console.log(qArray[i].qName); }
}



var dumpVars = function() {
  for (var i = 1 ; i < qArray.length ; i++) {
    console.log(t, qArray[i].qType, qArray[i].qName, qArray[i].j, qArray[i].k)
  }
}


var debugRun = function() {
// function debugRun () {
  initModel();
//  logData();
  for (var i = 1 ; i <= 100 ; i++) {
    warmupAuxen();
    warmupRates();
    tock();
//    logData();
  }
  while (t <= stopTime) {
    updateLevels();
//    if (t < 1904) { dumpVars(); console.log("****") };
    updateAuxen();
//    if (t == 1911) { dumpVars(); };
    updateRates();
//    if (t == 1911) { dumpVars(); };
    tock();
//    if (t == 1911) { dumpVars(); };
    t += dt;
    logData();
  }  
}



//THE POPULATION SECTOR

var population = new Aux("population", 1);
  population.units = "persons";
  population.plotColor = "#e07154";
  population.plotMin = 0;
  population.plotMax = 1.6e10;
  population.updateFn = function() {
    return population0To14.k +
           population15To44.k +
           population45To64.k +
           population65AndOver.k;
  }

var population0To14 = new Level("population0To14", 2, 6.5e8);
  population0To14.units = "persons";
  population0To14.updateFn = function() {
    return population0To14.j + dt * 
            (birthsPerYear.j - deathsPerYear0To14.j - maturationsPerYear14to15.j);
  }
  
var deathsPerYear0To14 = new Rate("deathsPerYear0To14", 3);
  deathsPerYear0To14.units = "persons per year";
  deathsPerYear0To14.updateFn = function() {
    return population0To14.k * mortality0To14.k;
  }

var mortality0To14 = new Table("mortality0To14", 4, [0.0567, 0.0366, 0.0243, 0.0155, 0.0082, 0.0023, 0.0010], 20, 80, 10);
  mortality0To14.units = "deaths per person-year";
  mortality0To14.dependencies = ["lifeExpectancy"];
  mortality0To14.updateFn = function() {
    return lifeExpectancy.k; 
  }

var maturationsPerYear14to15 = new Rate("maturationsPerYear14to15", 5);
  maturationsPerYear14to15.units = "persons per year";
  maturationsPerYear14to15.updateFn = function() {
    return population0To14.k * (1 - mortality0To14.k) / 15;
  }

var population15To44 = new Level("population15To44", 6, 7.0e8);
  population15To44.units = "persons";
  population15To44.updateFn = function() {
    return population15To44.j + dt * 
            (maturationsPerYear14to15.j - deathsPerYear15To44.j - maturationsPerYear44to45.j);
  }
  
var deathsPerYear15To44 = new Rate("deathsPerYear15To44", 7);
  deathsPerYear15To44.units = "persons per year";
  deathsPerYear15To44.updateFn = function() {
    return population15To44.k * mortality15To44.k;
  }

var mortality15To44 = new Table("mortality15To44", 8, [0.0266, 0.0171, 0.0110, 0.0065, 0.0040, 0.0016, 0.0008], 20, 80, 10);
  mortality15To44.units = "deaths per person-year";
  mortality15To44.dependencies = ["lifeExpectancy"];
  mortality15To44.updateFn = function() {
    return lifeExpectancy.k; 
  }

var maturationsPerYear44to45 = new Rate("maturationsPerYear44to45", 9);
  maturationsPerYear44to45.units = "persons per year";
  maturationsPerYear44to45.updateFn = function() {
    return population15To44.k * (1 - mortality15To44.k) / 30;
  }

var population45To64 = new Level("population45To64", 10, 1.9e8);
  population45To64.units = "persons";
  population45To64.updateFn = function() {
    return population45To64.j + dt * 
            (maturationsPerYear44to45.j - deathsPerYear45To64.j - maturationsPerYear64to65.j);
  }
  
var deathsPerYear45To64 = new Rate("deathsPerYear45To64", 11);
  deathsPerYear45To64.units = "persons per year";
  deathsPerYear45To64.updateFn = function() {
    return population45To64.k * mortality45To64.k;
  }

var mortality45To64 = new Table("mortality45To64", 12, [0.0562, 0.0373, 0.0252, 0.0171, 0.0118, 0.0083, 0.0060], 20, 80, 10);
  mortality45To64.units = "deaths per person-year";
  mortality45To64.dependencies = ["lifeExpectancy"];
  mortality45To64.updateFn = function() {
    return lifeExpectancy.k; 
  }

var maturationsPerYear64to65 = new Rate("maturationsPerYear64to65", 13);
  maturationsPerYear64to65.units = "persons per year";
  maturationsPerYear64to65.updateFn = function() {
    return population45To64.k * (1 - mortality45To64.k) / 20;
  }

var population65AndOver = new Level("population65AndOver", 14, 6.0e7);
  population65AndOver.units = "persons";
  population65AndOver.updateFn = function() {
    return population65AndOver.j + dt * 
            (maturationsPerYear64to65.j - deathsPerYear65AndOver.j);
  }
  
var deathsPerYear65AndOver = new Rate("deathsPerYear65AndOver", 15);
  deathsPerYear65AndOver.units = "persons per year";
  deathsPerYear65AndOver.updateFn = function() {
    return population65AndOver.k * mortality65AndOver.k;
  }

var mortality65AndOver = new Table("mortality65AndOver", 16, [0.13, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04], 20, 80, 10);
  mortality65AndOver.units = "deaths per person-year";
  mortality65AndOver.dependencies = ["lifeExpectancy"];
  mortality65AndOver.updateFn = function() {
    return lifeExpectancy.k; 
  }




// The Death-Rate Subsector

var deathsPerYear = new Aux("deathsPerYear", 17);
  deathsPerYear.units = "persons per year";
  deathsPerYear.updateFn = function() {
    return deathsPerYear0To14.j +
           deathsPerYear15To44.j +
           deathsPerYear45To64.j +
           deathsPerYear65AndOver.j;
  }

var crudeDeathRate = new Aux("crudeDeathRate", 18);
  crudeDeathRate.units = "deaths per 1000 person-years";
  crudeDeathRate.dependencies = ["deathsPerYear", "population"]
  crudeDeathRate.plotColor = "#650d99";
  crudeDeathRate.plotMin = 0;
  crudeDeathRate.plotMax = 50;
  crudeDeathRate.updateFn = function() {
    return 1000 * deathsPerYear.k / population.k;
  }

var lifeExpectancy = new Aux("lifeExpectancy", 19);
  lifeExpectancy.units = "years";
  lifeExpectancy.plotColor = "#666666";
  lifeExpectancy.plotMin = 0;
  lifeExpectancy.plotMax = 80;
  lifeExpectancy.dependencies = ["lifetimeMultiplierFromFood", "lifetimeMultiplierFromHealthServices", "lifetimeMultiplierFromPollution", "lifetimeMultiplierFromCrowding"]
  lifeExpectancy.normal = 32;
  lifeExpectancy.updateFn = function() {
    return lifeExpectancy.normal * 
           lifetimeMultiplierFromFood.k *
           lifetimeMultiplierFromHealthServices.k *
           lifetimeMultiplierFromPollution.k *
           lifetimeMultiplierFromCrowding.k;
  }

var subsistenceFoodPerCapitaK = 230;  // kilograms per person-year, used in eqns 20, 127

var lifetimeMultiplierFromFood = new Table("lifetimeMultiplierFromFood", 20, [0, 1, 1.2, 1.3, 1.35, 1.4], 0, 5, 1);
  lifetimeMultiplierFromFood.units = "dimensionless";
  lifetimeMultiplierFromFood.dependencies = ["foodPerCapita"];
  lifetimeMultiplierFromFood.updateFn = function() {
    return foodPerCapita.k / subsistenceFoodPerCapitaK;
  }
  
var healthServicesAllocationsPerCapita = new Table("healthServicesAllocationsPerCapita", 21, [0, 20, 50, 95, 140, 175, 200, 220, 230], 0, 2000, 250);
  healthServicesAllocationsPerCapita.units = "dollars per person-year";
  healthServicesAllocationsPerCapita.dependencies = ["serviceOutputPerCapita"];
  healthServicesAllocationsPerCapita.updateFn = function() {
    return serviceOutputPerCapita.k; 
  }

var effectiveHealthServicesPerCapitaImpactDelay = 20; // years, used in eqn 22

var effectiveHealthServicesPerCapita = new Smooth("effectiveHealthServicesPerCapita", 22, effectiveHealthServicesPerCapitaImpactDelay);
  effectiveHealthServicesPerCapita.units = "dollars per person-year";
  effectiveHealthServicesPerCapita.dependencies = ["healthServicesAllocationsPerCapita"];
  effectiveHealthServicesPerCapita.initFn = function() {
    return healthServicesAllocationsPerCapita; 
  }


var lifetimeMultiplierFromHealthServices = new Aux("lifetimeMultiplierFromHealthServices", 23);
  lifetimeMultiplierFromHealthServices.units = "dimensionless";
  lifetimeMultiplierFromHealthServices.dependencies = ["lifetimeMultiplierFromHealthServicesBefore", "lifetimeMultiplierFromHealthServicesAfter"];
  lifetimeMultiplierFromHealthServices.policyYear = 1940;
  lifetimeMultiplierFromHealthServices.updateFn = function() {
    return clip(lifetimeMultiplierFromHealthServicesAfter.k,
                lifetimeMultiplierFromHealthServicesBefore.k,
                t, 
                lifetimeMultiplierFromHealthServices.policyYear);
  }
  
var lifetimeMultiplierFromHealthServicesBefore = new Table("lifetimeMultiplierFromHealthServicesBefore", 24, [1, 1.1, 1.4, 1.6, 1.7, 1.8], 0, 100, 20);
  lifetimeMultiplierFromHealthServicesBefore.units = "dimensionless";
  lifetimeMultiplierFromHealthServicesBefore.dependencies = ["effectiveHealthServicesPerCapita"];
  lifetimeMultiplierFromHealthServicesBefore.updateFn = function() {
    return effectiveHealthServicesPerCapita.k;
  }
  
var lifetimeMultiplierFromHealthServicesAfter = new Table("lifetimeMultiplierFromHealthServicesAfter", 25, [1, 1.4, 1.6, 1.8, 1.95, 2.0], 0, 100, 20);
  lifetimeMultiplierFromHealthServicesAfter.units = "dimensionless";
  lifetimeMultiplierFromHealthServicesAfter.dependencies = ["effectiveHealthServicesPerCapita"];
  lifetimeMultiplierFromHealthServicesAfter.updateFn = function() {
    return effectiveHealthServicesPerCapita.k;
  }
  
var fractionOfPopulationUrban = new Table("fractionOfPopulationUrban", 26, [0, 0.2, 0.4, 0.5, 0.58, 0.65, 0.72, 0.78, 0.80], 0, 1.6e10, 2.0e9);
  fractionOfPopulationUrban.units = "dimensionless";
  fractionOfPopulationUrban.dependencies = ["population"];
  fractionOfPopulationUrban.updateFn = function() {
    return population.k;
  }

var crowdingMultiplierFromIndustrialization = new Table("crowdingMultiplierFromIndustrialization", 27, [0.5, 0.05, -0.1, -0.08, -0.02, 0.05, 0.1, 0.15, 0.2],
     0, 1600, 200);
  crowdingMultiplierFromIndustrialization.units = "dimensionless";
  crowdingMultiplierFromIndustrialization.dependencies = ["industrialOutputPerCapita"];
  crowdingMultiplierFromIndustrialization.updateFn = function() {
    return industrialOutputPerCapita.k;
  }

var lifetimeMultiplierFromCrowding = new Aux("lifetimeMultiplierFromCrowding", 28);
  lifetimeMultiplierFromCrowding.units = "dimensionless";
  lifetimeMultiplierFromCrowding.updateFn = function() {
    return 1 - (crowdingMultiplierFromIndustrialization.k * fractionOfPopulationUrban.k); 
  }

var lifetimeMultiplierFromPollution 
      = new Table("lifetimeMultiplierFromPollution", 29, [1.0, 0.99, 0.97, 0.95, 0.90, 0.85, 0.75, 0.65, 0.55, 0.40, 0.20], 0, 100, 10);
  lifetimeMultiplierFromPollution.units = "dimensionless";
  lifetimeMultiplierFromPollution.dependencies = ["indexOfPersistentPollution"];
  lifetimeMultiplierFromPollution.updateFn = function() {
    return indexOfPersistentPollution.k;
  }


// The Birth-Rate Subsector


var birthsPerYear = new Rate("birthsPerYear", 30);
  birthsPerYear.units = "persons per year";
  birthsPerYear.plotThisVar = true;
  birthsPerYear.reproductiveLifetime = 30;          // years
  birthsPerYear.populationEquilibriumTime = 4000;   // year
  birthsPerYear.updateFn = function() {
    var after = deathsPerYear.k;
    var before = totalFertility.k * population15To44.k * 0.5 / birthsPerYear.reproductiveLifetime;
    return clip(after, before, t, birthsPerYear.populationEquilibriumTime);
  }
  
var crudeBirthRate = new Aux("crudeBirthRate", 31);
  crudeBirthRate.units = "births per 1000 person-years";
  crudeBirthRate.dependencies = ["population"]
  crudeBirthRate.plotColor = "#f6f648";
  crudeBirthRate.plotMin = 0;
  crudeBirthRate.plotMax = 50;
  crudeBirthRate.updateFn = function() {
    return 1000 * birthsPerYear.j / population.k;
  }

var totalFertility = new Aux("totalFertility", 32);
  totalFertility.units = "dimensionless";
  totalFertility.dependencies = ["maxTotalFertility", "fertilityControlEffectiveness", "desiredTotalFertility"]
  totalFertility.updateFn = function() {
    return Math.min(maxTotalFertility.k,
                   (maxTotalFertility.k * (1 - fertilityControlEffectiveness.k) +
                        desiredTotalFertility.k * fertilityControlEffectiveness.k));
  }
  
var maxTotalFertility = new Aux("maxTotalFertility", 33);
  maxTotalFertility.units = "dimensionless";
  maxTotalFertility.dependencies = ["fecundityMultiplier"]
  maxTotalFertility.normal = 12;   // dimensionless
  maxTotalFertility.updateFn = function() {
    return maxTotalFertility.normal * fecundityMultiplier.k;
  }

var fecundityMultiplier = new Table("fecundityMultiplier", 34, [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.05, 1.1], 0, 80, 10);
  fecundityMultiplier.units = "dimensionless";
  fecundityMultiplier.dependencies = ["lifeExpectancy"];
  fecundityMultiplier.updateFn = function() {
    return lifeExpectancy.k;
  }
 
var desiredTotalFertility = new Aux("desiredTotalFertility", 35);
  desiredTotalFertility.units = "dimensionless";
  desiredTotalFertility.dependencies = [ "desiredCompletedFamilySize", "compensatoryMultiplierFromPerceivedLifeExpectancy" ]
  desiredTotalFertility.updateFn = function() {
    return desiredCompletedFamilySize.k * compensatoryMultiplierFromPerceivedLifeExpectancy.k;
  }
  
var compensatoryMultiplierFromPerceivedLifeExpectancy = 
        new Table("compensatoryMultiplierFromPerceivedLifeExpectancy", 36, [3.0, 2.1, 1.6, 1.4, 1.3, 1.2, 1.1, 1.05, 1.0], 0, 80, 10);
  compensatoryMultiplierFromPerceivedLifeExpectancy.units = "dimensionless";
  compensatoryMultiplierFromPerceivedLifeExpectancy.dependencies = ["perceivedLifeExpectancy"];
  compensatoryMultiplierFromPerceivedLifeExpectancy.updateFn = function() {
    return perceivedLifeExpectancy.k;
  }

var lifetimePerceptionDelayK = 20;      // years, used in eqn 37

var perceivedLifeExpectancy = new Delay3("perceivedLifeExpectancy", 37, lifetimePerceptionDelayK);
  perceivedLifeExpectancy.units = "years";
  perceivedLifeExpectancy.dependencies = ["lifeExpectancy"];
  perceivedLifeExpectancy.initFn = function() { return lifeExpectancy; }


var desiredCompletedFamilySize = new Aux("desiredCompletedFamilySize", 38);
  desiredCompletedFamilySize.units = "dimensionless";            // not persons?
  desiredCompletedFamilySize.dependencies = ["familyResponseToSocialNorm", "socialFamilySizeNorm"];
  desiredCompletedFamilySize.normal = 4.0;
  zeroPopulationGrowthTargetYear = 4000;
  desiredCompletedFamilySize.updateFn = function() {
    return clip(2.0, (desiredCompletedFamilySize.normal * familyResponseToSocialNorm.k * socialFamilySizeNorm.k), t, zeroPopulationGrowthTargetYear);
  }

var socialFamilySizeNorm = new Table("socialFamilySizeNorm", 39, [1.25, 1, 0.9, 0.8, 0.75], 0, 800, 200);
  socialFamilySizeNorm.units = "dimensionless";
  socialFamilySizeNorm.dependencies = ["delayedIndustrialOutputPerCapita"];
  socialFamilySizeNorm.updateFn = function() {
    return delayedIndustrialOutputPerCapita.k;
  }

var socialAdjustmentDelayK = 20;    // years, used in eqn 40

var delayedIndustrialOutputPerCapita = new Delay3("delayedIndustrialOutputPerCapita", 40, socialAdjustmentDelayK);
  delayedIndustrialOutputPerCapita.units = "dollars per person-year";
  delayedIndustrialOutputPerCapita.dependencies = ["industrialOutputPerCapita"];
  delayedIndustrialOutputPerCapita.initFn = function() { return industrialOutputPerCapita; }

  
var familyResponseToSocialNorm = new Table("familyResponseToSocialNorm", 41, [0.5, 0.6, 0.7, 0.85, 1.0], -0.2, 0.2, 0.1);
  familyResponseToSocialNorm.units = "dimensionless";
  familyResponseToSocialNorm.dependencies = ["familyIncomeExpectation"];
  familyResponseToSocialNorm.updateFn = function() {
    return familyIncomeExpectation.k;
  }
  
var familyIncomeExpectation = new Aux("familyIncomeExpectation", 42);
  familyIncomeExpectation.units = "dimensionless";
  familyIncomeExpectation.dependencies = ["industrialOutputPerCapita", "averageIndustrialOutputPerCapita"];
  familyIncomeExpectation.updateFn = function() {
    return (industrialOutputPerCapita.k - averageIndustrialOutputPerCapita.k) / averageIndustrialOutputPerCapita.k;
  }
  
var incomeExpectationAveragingTimeK = 3; // years, used in eqn 43
  
var averageIndustrialOutputPerCapita = new Smooth("averageIndustrialOutputPerCapita", 43, incomeExpectationAveragingTimeK);
  averageIndustrialOutputPerCapita.units = "dollars per person-year";
  averageIndustrialOutputPerCapita.dependencies = ["industrialOutputPerCapita"];
  averageIndustrialOutputPerCapita.initFn = function() { return industrialOutputPerCapita; }


var needForFertilityControl = new Aux("needForFertilityControl", 44);
  needForFertilityControl.units = "dimensionless";
  needForFertilityControl.dependencies = ["maxTotalFertility", "desiredTotalFertility"];
  needForFertilityControl.updateFn = function() {
    return (maxTotalFertility.k / desiredTotalFertility.k) - 1;
  }
  
var fertilityControlEffectiveness = new Table("fertilityControlEffectiveness", 45, [0.75, 0.85, 0.90, 0.95, 0.98, 0.99, 1.0], 0, 3, 0.5 );
  fertilityControlEffectiveness.units = "dimensionless";
  fertilityControlEffectiveness.dependencies = ["fertilityControlFacilitiesPerCapita"];
  fertilityControlEffectiveness.updateFn = function() {
    return fertilityControlFacilitiesPerCapita.k;
  }

var healthServicesImpactDelayK = 20;    // years, for eqn 46

var fertilityControlFacilitiesPerCapita = new Delay3("fertilityControlFacilitiesPerCapita", 46, healthServicesImpactDelayK);
  fertilityControlFacilitiesPerCapita.units = "dollars per person-year";
  fertilityControlFacilitiesPerCapita.dependencies = ["fertilityControlAllocationPerCapita"];
  fertilityControlFacilitiesPerCapita.initFn = function() { return fertilityControlAllocationPerCapita; }

  
var fertilityControlAllocationPerCapita = new Aux("fertilityControlAllocationPerCapita", 47);
  fertilityControlAllocationPerCapita.units = "dollars per person-year";
  fertilityControlAllocationPerCapita.dependencies = ["serviceOutputPerCapita", "fractionOfServicesAllocatedToFertilityControl"];
  fertilityControlAllocationPerCapita.updateFn = function() {
    return fractionOfServicesAllocatedToFertilityControl.k * serviceOutputPerCapita.k;
  }
  
var fractionOfServicesAllocatedToFertilityControl = new Table("fractionOfServicesAllocatedToFertilityControl", 48, [0.0, 0.005, 0.015, 0.025, 0.030, 0.035], 0, 10, 2);
  fractionOfServicesAllocatedToFertilityControl.units = "dimensionless";
  fractionOfServicesAllocatedToFertilityControl.dependencies = ["needForFertilityControl"];
  fractionOfServicesAllocatedToFertilityControl.updateFn = function() {
    return needForFertilityControl.k;
  }
  



// THE CAPITAL SECTOR

// The Industrial Subsector


var industrialOutputPerCapita = new Aux("industrialOutputPerCapita", 49);
  industrialOutputPerCapita.units = "dollars per person-year";
  industrialOutputPerCapita.dependencies = ["industrialOutput", "population"];
  industrialOutputPerCapita.plotColor = "#4a6892";
  industrialOutputPerCapita.plotMin = 0;
  industrialOutputPerCapita.plotMax = 500;
  industrialOutputPerCapita.updateFn = function() {
    return industrialOutput.k / population.k;
  }
  
var industrialOutput = new Aux("industrialOutput", 50);
  industrialOutput.units = "dollars per year";
  industrialOutput.valueIn1970 = 7.9e11;   // for eqns 106 and 107
  industrialOutput.dependencies = ["fractionOfCapitalAllocatedToObtainingResources", "capitalUtilizationFraction", "industrialCapitalOutputRatio"];
  industrialOutput.updateFn = function() {
    return industrialCapital.k * (1 - fractionOfCapitalAllocatedToObtainingResources.k) * capitalUtilizationFraction.k / industrialCapitalOutputRatio.k;
  }
  
var industrialCapitalOutputRatio = new Aux("industrialCapitalOutputRatio", 51);
  industrialCapitalOutputRatio.units = "years";
  industrialCapitalOutputRatio.before = 3;
  industrialCapitalOutputRatio.after = 3;
  industrialCapitalOutputRatio.updateFn = function() {
    return clip(industrialCapitalOutputRatio.after, industrialCapitalOutputRatio.before, t, policyYear);
  }

var industrialCapital = new Level("industrialCapital", 52, 2.1e11);
  industrialCapital.units = "dollars";
  industrialCapital.updateFn = function() {
    return industrialCapital.j + dt * 
            (industrialCapitalInvestmentRate.j - industrialCapitalDepreciationRate.j);
  }
  
var industrialCapitalDepreciationRate = new Rate("industrialCapitalDepreciationRate", 53);
  industrialCapitalDepreciationRate.units = "dollars per year";
  industrialCapitalDepreciationRate.updateFn = function() {
    return industrialCapital.k / averageLifetimeOfIndustrialCapital.k;
  }
  
var averageLifetimeOfIndustrialCapital = new Aux("averageLifetimeOfIndustrialCapital", 54);
  averageLifetimeOfIndustrialCapital.units = "years";
  averageLifetimeOfIndustrialCapital.before = 14;
  averageLifetimeOfIndustrialCapital.after = 14;
  averageLifetimeOfIndustrialCapital.updateFn = function() {
    return clip(averageLifetimeOfIndustrialCapital.after, averageLifetimeOfIndustrialCapital.before, t, policyYear);
  }
  
var industrialCapitalInvestmentRate = new Rate("industrialCapitalInvestmentRate", 55);
  industrialCapitalInvestmentRate.units = "dollars per year";
  industrialCapitalInvestmentRate.updateFn = function() {
    return industrialOutput.k * fractionOfIndustrialOutputAllocatedToIndustry.k;
  }
  
var fractionOfIndustrialOutputAllocatedToIndustry = new Aux("fractionOfIndustrialOutputAllocatedToIndustry", 56);
  fractionOfIndustrialOutputAllocatedToIndustry.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToIndustry.dependencies = ["fractionOfIndustrialOutputAllocatedToAgriculture", "fractionOfIndustrialOutputAllocatedToServices", "fractionOfIndustrialOutputAllocatedToConsumption"];
  fractionOfIndustrialOutputAllocatedToIndustry.updateFn = function() {
    return 1 - fractionOfIndustrialOutputAllocatedToAgriculture.k - fractionOfIndustrialOutputAllocatedToServices.k - fractionOfIndustrialOutputAllocatedToConsumption.k;
  }
  
var fractionOfIndustrialOutputAllocatedToConsumption = new Aux("fractionOfIndustrialOutputAllocatedToConsumption", 57);
  fractionOfIndustrialOutputAllocatedToConsumption.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToConsumption.dependencies = ["fractionOfIndustrialOutputAllocatedToConsumptionVariable"];
  fractionOfIndustrialOutputAllocatedToConsumption.industrialEquilibriumTime = 4000;  // year
  fractionOfIndustrialOutputAllocatedToConsumption.updateFn = function() {
    return clip(fractionOfIndustrialOutputAllocatedToConsumptionVariable.k, fractionOfIndustrialOutputAllocatedToConsumptionConstant.k, t, fractionOfIndustrialOutputAllocatedToConsumption.industrialEquilibriumTime);
  }
  
var fractionOfIndustrialOutputAllocatedToConsumptionConstant = new Aux("fractionOfIndustrialOutputAllocatedToConsumptionConstant", 58);
  fractionOfIndustrialOutputAllocatedToConsumptionConstant.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToConsumptionConstant.before = 0.43;
  fractionOfIndustrialOutputAllocatedToConsumptionConstant.after = 0.43;
  fractionOfIndustrialOutputAllocatedToConsumptionConstant.updateFn = function() {
    return clip(fractionOfIndustrialOutputAllocatedToConsumptionConstant.after, fractionOfIndustrialOutputAllocatedToConsumptionConstant.before, t, policyYear);
  }
  
var fractionOfIndustrialOutputAllocatedToConsumptionVariable = new Table("fractionOfIndustrialOutputAllocatedToConsumptionVariable", 59, [0.3, 0.32, 0.34, 0.36, 0.38, 0.43, 0.73, 0.77, 0.81, 0.82, 0.83], 0, 2, 0.2);
  fractionOfIndustrialOutputAllocatedToConsumptionVariable.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToConsumptionVariable.dependencies = ["industrialOutputPerCapita"];
  fractionOfIndustrialOutputAllocatedToConsumptionVariable.industrialOutputPerCapitaDesired = 400;
  fractionOfIndustrialOutputAllocatedToConsumptionVariable.updateFn = function() {
    return industrialOutputPerCapita.k / fractionOfIndustrialOutputAllocatedToConsumptionVariable.industrialOutputPerCapitaDesired;
  }
  

// The Service Subsector

var indicatedServiceOutputPerCapita = new Aux("indicatedServiceOutputPerCapita", 60);
  indicatedServiceOutputPerCapita.units = "dollars per person-year";
  indicatedServiceOutputPerCapita.dependencies = ["indicatedServiceOutputPerCapitaAfter", "indicatedServiceOutputPerCapitaBefore"];
  indicatedServiceOutputPerCapita.updateFn = function() {
    return clip(indicatedServiceOutputPerCapitaAfter.k, indicatedServiceOutputPerCapitaBefore.k, t, policyYear);
  }
  
var indicatedServiceOutputPerCapitaBefore = new Table("indicatedServiceOutputPerCapitaBefore", 61, [40, 300, 640, 1000, 1220, 1450, 1650, 1800, 2000], 0, 1600, 200);
  indicatedServiceOutputPerCapitaBefore.units = "dollars per person-year";
  indicatedServiceOutputPerCapitaBefore.dependencies = ["industrialOutputPerCapita"];
  indicatedServiceOutputPerCapitaBefore.updateFn = function() {
    return industrialOutputPerCapita.k;
  }
  
var indicatedServiceOutputPerCapitaAfter = new Table("indicatedServiceOutputPerCapitaAfter", 62, [40, 300, 640, 1000, 1220, 1450, 1650, 1800, 2000], 0, 1600, 200);
  indicatedServiceOutputPerCapitaAfter.units = "dollars per person-year";
  indicatedServiceOutputPerCapitaAfter.dependencies = ["industrialOutputPerCapita"];
  indicatedServiceOutputPerCapitaAfter.updateFn = function() {
    return industrialOutputPerCapita.k;
  }
  
var fractionOfIndustrialOutputAllocatedToServices = new Aux("fractionOfIndustrialOutputAllocatedToServices", 63);
  fractionOfIndustrialOutputAllocatedToServices.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToServices.dependencies = ["fractionOfIndustrialOutputAllocatedToServicesBefore", "fractionOfIndustrialOutputAllocatedToServicesAfter"];
  fractionOfIndustrialOutputAllocatedToServices.updateFn = function() {
    return clip(fractionOfIndustrialOutputAllocatedToServicesAfter.k, fractionOfIndustrialOutputAllocatedToServicesBefore.k, t, policyYear);
  }
  
var fractionOfIndustrialOutputAllocatedToServicesBefore = new Table("fractionOfIndustrialOutputAllocatedToServicesBefore", 64, [0.3, 0.2, 0.1, 0.05, 0], 0, 2, 0.5);
  fractionOfIndustrialOutputAllocatedToServicesBefore.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToServicesBefore.dependencies = ["serviceOutputPerCapita", "indicatedServiceOutputPerCapita"];
  fractionOfIndustrialOutputAllocatedToServicesBefore.updateFn = function() {
    return serviceOutputPerCapita.k / indicatedServiceOutputPerCapita.k;
  }
  
var fractionOfIndustrialOutputAllocatedToServicesAfter = new Table("fractionOfIndustrialOutputAllocatedToServicesAfter", 65, [0.3, 0.2, 0.1, 0.05, 0], 0, 2, 0.5);
  fractionOfIndustrialOutputAllocatedToServicesAfter.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToServicesAfter.dependencies = ["serviceOutputPerCapita", "indicatedServiceOutputPerCapita"];
  fractionOfIndustrialOutputAllocatedToServicesAfter.updateFn = function() {
    return serviceOutputPerCapita.k / indicatedServiceOutputPerCapita.k;
  }
  
var serviceCapitalInvestmentRate = new Rate("serviceCapitalInvestmentRate", 66);
  serviceCapitalInvestmentRate.units = "dollars per year";
  serviceCapitalInvestmentRate.updateFn = function() {
    return industrialOutput.k * fractionOfIndustrialOutputAllocatedToServices.k;
  }
  
var serviceCapital = new Level("serviceCapital", 67, 1.44e11);
  serviceCapital.units = "dollars";
  serviceCapital.updateFn = function() {
    return serviceCapital.j + dt * 
            (serviceCapitalInvestmentRate.j - serviceCapitalDepreciationRate.j);
  }
  
var serviceCapitalDepreciationRate = new Rate("serviceCapitalDepreciationRate", 68);
  serviceCapitalDepreciationRate.units = "dollars per year";
  serviceCapitalDepreciationRate.updateFn = function() {
    return serviceCapital.k / averageLifetimeOfServiceCapital.k;
  }
  
var averageLifetimeOfServiceCapital = new Aux("averageLifetimeOfServiceCapital", 69);
  averageLifetimeOfServiceCapital.units = "years";
  averageLifetimeOfServiceCapital.before = 20;   // years
  averageLifetimeOfServiceCapital.after = 20;    // years
  averageLifetimeOfServiceCapital.updateFn = function() {
    return clip(averageLifetimeOfServiceCapital.after, averageLifetimeOfServiceCapital.before, t, policyYear);
  }
  
var serviceOutput = new Aux("serviceOutput", 70);
  serviceOutput.units = "dollars per year";
  serviceOutput.plotColor = "#4a8a91";
  serviceOutput.plotMin = 0;
  serviceOutput.plotMax = 1.0e13;
  serviceOutput.dependencies = ["capitalUtilizationFraction", "serviceCapitalOutputRatio"];
  serviceOutput.updateFn = function() {
    return (serviceCapital.k * capitalUtilizationFraction.k) / serviceCapitalOutputRatio.k;
  }

var serviceOutputPerCapita = new Aux("serviceOutputPerCapita", 71);
  serviceOutputPerCapita.units = "dollars per person-year";
  serviceOutputPerCapita.dependencies = ["serviceOutput", "population"];
  serviceOutputPerCapita.updateFn = function() {
    return serviceOutput.k / population.k;
  }

var serviceCapitalOutputRatio = new Aux("serviceCapitalOutputRatio", 72);
  serviceCapitalOutputRatio.units = "years";
  serviceCapitalOutputRatio.before = 1;
  serviceCapitalOutputRatio.after = 1;
  serviceCapitalOutputRatio.updateFn = function() {
    return clip(serviceCapitalOutputRatio.after, serviceCapitalOutputRatio.before, t, policyYear);
  }


// The Jobs Subsector

var jobs = new Aux("jobs", 73)
  jobs.units = "persons";
  jobs.dependencies = ["potentialJobsInIndustrialSector", "potentialJobsInAgriculturalSector", "potentialJobsInServiceSector"];
  jobs.updateFn = function() {
    return potentialJobsInIndustrialSector.k + potentialJobsInAgriculturalSector.k + potentialJobsInServiceSector.k;
  }

var potentialJobsInIndustrialSector = new Aux("potentialJobsInIndustrialSector", 74);
  potentialJobsInIndustrialSector.units = "persons";
  potentialJobsInIndustrialSector.dependencies = ["jobsPerIndustrialCapitalUnit"];
  potentialJobsInIndustrialSector.updateFn = function() {
    return industrialCapital.k * jobsPerIndustrialCapitalUnit.k;
  }

var jobsPerIndustrialCapitalUnit = new Table("jobsPerIndustrialCapitalUnit", 75, [0.00037, 0.00018, 0.00012, 0.00009, 0.00007, 0.00006], 50, 800, 150);
  jobsPerIndustrialCapitalUnit.units = "persons per dollar";
  jobsPerIndustrialCapitalUnit.dependencies = ["industrialOutputPerCapita"];
  jobsPerIndustrialCapitalUnit.updateFn = function() {
    return industrialOutputPerCapita.k; 
  }
  
var potentialJobsInServiceSector = new Aux("potentialJobsInServiceSector", 76);
  potentialJobsInServiceSector.units = "persons";
  potentialJobsInServiceSector.dependencies = ["jobsPerServiceCapitalUnit"];
  potentialJobsInServiceSector.updateFn = function() {
    return serviceCapital.k * jobsPerServiceCapitalUnit.k;
  }

var jobsPerServiceCapitalUnit = new Table("jobsPerServiceCapitalUnit", 77, [.0011, 0.0006, 0.00035, 0.0002, 0.00015, 0.00015], 50, 800, 150);
  jobsPerServiceCapitalUnit.units = "persons per dollar";
  jobsPerServiceCapitalUnit.dependencies = ["serviceOutputPerCapita"];
  jobsPerServiceCapitalUnit.updateFn = function() {
    return serviceOutputPerCapita.k; 
  }
  
var potentialJobsInAgriculturalSector = new Aux("potentialJobsInAgriculturalSector", 78);
  potentialJobsInAgriculturalSector.units = "persons";
  potentialJobsInAgriculturalSector.dependencies = ["jobsPerHectare"];
  potentialJobsInAgriculturalSector.updateFn = function() {
    return arableLand.k * jobsPerHectare.k;
  }

var jobsPerHectare = new Table("jobsPerHectare", 79, [2, 0.5, 0.4, 0.3, 0.27, 0.24, 0.2, 0.2], 2, 30, 4);
  jobsPerHectare.units = "persons per hectare";
  jobsPerHectare.dependencies = ["agriculturalInputsPerHectare"];
  jobsPerHectare.updateFn = function() {
    return agriculturalInputsPerHectare.k;
  }
  
var laborForce = new Aux("laborForce", 80);
  laborForce.units = "persons";
  laborForce.participationFraction = 0.75  // dimensionless
  laborForce.updateFn = function() {
    return (population15To44.k + population45To64.k) * laborForce.participationFraction;
  }

var laborUtilizationFraction = new Aux("laborUtilizationFraction", 81);
  laborUtilizationFraction.units = "dimensionless";
  laborUtilizationFraction.dependencies = ["jobs", "laborForce"];
  laborUtilizationFraction.updateFn = function() {
    return jobs.k / laborForce.k;
  }

var laborUtilizationFractionDelayedDelayTime = 2   // years, eqn 82

var laborUtilizationFractionDelayed = new Smooth("laborUtilizationFractionDelayed", 82, laborUtilizationFractionDelayedDelayTime);
  laborUtilizationFractionDelayed.units = "dimensionless";
  laborUtilizationFractionDelayed.dependencies = ["laborUtilizationFraction"];
  laborUtilizationFractionDelayed.initFn = function() { return laborUtilizationFraction; }

var capitalUtilizationFraction = new Table("capitalUtilizationFraction", 83, [1.0, 0.9, 0.7, 0.3, 0.1, 0.1], 1, 11, 2);
  capitalUtilizationFraction.units = "dimensionless";
  capitalUtilizationFraction.dependencies = [];   // "laborUtilizationFractionDelayed" removed to break cycle
  capitalUtilizationFraction.updateFn = function() {
    return laborUtilizationFractionDelayed.k || 1.0;   // to break circularity
  }


// THE AGRICULTURAL SECTOR

// Loop 1: Food from Investment in Land Development

var landFractionCultivated = new Aux("landFractionCultivated", 84);
  landFractionCultivated.units = "dimensionless";
  landFractionCultivated.potentiallyArableLandTotal = 3.2e9   // hectares, used here and in eqn 97
  landFractionCultivated.updateFn = function() {
    return arableLand.k / landFractionCultivated.potentiallyArableLandTotal;
  }

var arableLand = new Level("arableLand", 85, 0.9e9);
  arableLand.units = "hectares";
  arableLand.plotColor = "#513210"
  arableLand.plotMin = 0;
  arableLand.plotMax = 3.0e9;
  arableLand.updateFn = function() {
    return arableLand.j + dt * 
            (landDevelopmentRate.j - landErosionRate.j - landRemovalForUrbanIndustrialUse.j);
  }
  
var potentiallyArableLand = new Level("potentiallyArableLand", 86, 2.3e9);
  potentiallyArableLand.units = "hectares";
  potentiallyArableLand.updateFn = function() {
    return potentiallyArableLand.j + dt * (-landDevelopmentRate.j)
  }
  
var food = new Aux("food", 87);
  food.units = "kilograms per year";
  food.dependencies = ["landYield"];
  food.landFractionHarvestedK = 0.7;   // dimensionless
  food.processingLossK = 0.1;          // dimensionless
  food.updateFn = function() {
    return landYield.k * arableLand.k * food.landFractionHarvestedK * (1 - food.processingLossK);
  }
  
var foodPerCapita = new Aux("foodPerCapita", 88);
  foodPerCapita.units = "kilograms per person-year";
  foodPerCapita.dependencies = ["food", "population"];
  foodPerCapita.plotColor = "#a8c3a5";
  foodPerCapita.plotMin = 0;
  foodPerCapita.plotMax = 1000;
  foodPerCapita.updateFn = function() {
    return food.k / population.k;
  }
  
var indicatedFoodPerCapita = new Aux("indicatedFoodPerCapita", 89);
  indicatedFoodPerCapita.units = "kilograms per person-year";
  indicatedFoodPerCapita.dependencies = ["indicatedFoodPerCapitaBefore", "indicatedFoodPerCapitaAfter"];
  indicatedFoodPerCapita.updateFn = function() {
    return clip(indicatedFoodPerCapitaAfter.k, indicatedFoodPerCapitaBefore.k, t, policyYear);
  }
  
var indicatedFoodPerCapitaBefore = new Table("indicatedFoodPerCapitaBefore", 90, [230, 480, 690, 850, 970, 1070, 1150, 1210, 1250], 0, 1600, 200)
  indicatedFoodPerCapitaBefore.units = "kilograms per person-year";
  indicatedFoodPerCapitaBefore.dependencies = ["industrialOutputPerCapita"];
  indicatedFoodPerCapitaBefore.updateFn = function() {
    return industrialOutputPerCapita.k;
  }
  
var indicatedFoodPerCapitaAfter = new Table("indicatedFoodPerCapitaAfter", 91, [230, 480, 690, 850, 970, 1070, 1150, 1210, 1250], 0, 1600, 200)
  indicatedFoodPerCapitaAfter.units = "kilograms per person-year";
  indicatedFoodPerCapitaAfter.dependencies = ["industrialOutputPerCapita"];
  indicatedFoodPerCapitaAfter.updateFn = function() {
    return industrialOutputPerCapita.k;
  }
  
var totalAgriculturalInvestment = new Aux("totalAgriculturalInvestment", 92);
  totalAgriculturalInvestment.units = "dollars per year";
  totalAgriculturalInvestment.dependencies = ["industrialOutput", "fractionOfIndustrialOutputAllocatedToAgriculture"];
  totalAgriculturalInvestment.updateFn = function() {
    return industrialOutput.k * fractionOfIndustrialOutputAllocatedToAgriculture.k;
  }
  
var fractionOfIndustrialOutputAllocatedToAgriculture = new Aux("fractionOfIndustrialOutputAllocatedToAgriculture", 93);
  fractionOfIndustrialOutputAllocatedToAgriculture.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToAgriculture.dependencies = ["fractionOfIndustrialOutputAllocatedToAgricultureBefore", "fractionOfIndustrialOutputAllocatedToAgricultureAfter"];
  fractionOfIndustrialOutputAllocatedToAgriculture.updateFn = function() {
    return clip(fractionOfIndustrialOutputAllocatedToAgricultureAfter.k, fractionOfIndustrialOutputAllocatedToAgricultureBefore.k, t, policyYear);
  }
  
var fractionOfIndustrialOutputAllocatedToAgricultureBefore = new Table("fractionOfIndustrialOutputAllocatedToAgricultureBefore", 94, [0.4, 0.2, 0.1, 0.025, 0, 0], 0, 2.5, 0.5)
  fractionOfIndustrialOutputAllocatedToAgricultureBefore.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToAgricultureBefore.dependencies = ["foodPerCapita", "indicatedFoodPerCapita"];
  fractionOfIndustrialOutputAllocatedToAgricultureBefore.updateFn = function() {
    return foodPerCapita.k / indicatedFoodPerCapita.k;
  }
  
var fractionOfIndustrialOutputAllocatedToAgricultureAfter = new Table("fractionOfIndustrialOutputAllocatedToAgricultureAfter", 95, [0.4, 0.2, 0.1, 0.025, 0, 0], 0, 2.5, 0.5)
  fractionOfIndustrialOutputAllocatedToAgricultureAfter.units = "dimensionless";
  fractionOfIndustrialOutputAllocatedToAgricultureAfter.dependencies = ["foodPerCapita", "indicatedFoodPerCapita"];
  fractionOfIndustrialOutputAllocatedToAgricultureAfter.updateFn = function() {
    return foodPerCapita.k / indicatedFoodPerCapita.k;
  }
  
var landDevelopmentRate = new Rate("landDevelopmentRate", 96);
  landDevelopmentRate.units = "hectares per year";
  landDevelopmentRate.updateFn = function() {
    return totalAgriculturalInvestment.k * fractionOfInputsAllocatedToLandDevelopment.k / developmentCostPerHectare.k;
  }
  
var developmentCostPerHectare = new Table("developmentCostPerHectare", 97, [100000, 7400, 5200, 3500, 2400, 1500, 750, 300, 150, 75, 50], 0, 1.0, 0.1)
  developmentCostPerHectare.units = "dollars per hectare";
  developmentCostPerHectare.updateFn = function() {
    return potentiallyArableLand.k / landFractionCultivated.potentiallyArableLandTotal;
  }
  

// Loop 2: Food from Investment in Agricultural Inputs


var currentAgriculturalInputs = new Aux("currentAgriculturalInputs", 98);
  currentAgriculturalInputs.units = "dollars per year";
  currentAgriculturalInputs.dependencies = ["totalAgriculturalInvestment", "fractionOfInputsAllocatedToLandDevelopment"];
  currentAgriculturalInputs.updateFn = function() {
    return totalAgriculturalInvestment.k * (1 - fractionOfInputsAllocatedToLandDevelopment.k);
  }

var averageLifetimeOfAgriculturalInputsK = 2; // years, eqn 99 (in lieu of 100)

var agriculturalInputs = new Smooth("agriculturalInputs", 99, averageLifetimeOfAgriculturalInputsK);
  agriculturalInputs.units = "dollars per year";
  agriculturalInputs.dependencies = [];   // "currentAgriculturalInputs" removed to break cycle
  agriculturalInputs.initFn = function() { return currentAgriculturalInputs; }
  agriculturalInputs.initVal = 5.0e9;

/*
var agriculturalInputs = new Smooth("agriculturalInputs", 99, averageLifetimeOfAgriculturalInputsK);
  agriculturalInputs.units = "dollars per year";
  agriculturalInputs.dependencies = [];   // "currentAgriculturalInputs" removed to break cycle
  agriculturalInputs.initFn = function() { return currentAgriculturalInputs; }
  agriculturalInputs.initVal = 5.0e9;
   = function() {
    agriculturalInputs.theInput = agriculturalInputs.initFn;
    agriculturalInputs.j = agriculturalInputs.k = 5.0e9;
  }
  agriculturalInputs.update = function() {
    if (agriculturalInputs.firstCall) {
      agriculturalInputs.firstCall = false;
      return agriculturalInputs.k;
    }
    else {
      agriculturalInputs.k = agriculturalInputs.j + dt * (agriculturalInputs.theInput.j - agriculturalInputs.j) / agriculturalInputs.del;
      return agriculturalInputs.k;
    }
  }
*/


// note: output of this equation goes unused
var averageLifetimeOfAgriculturalInputs = new Aux("averageLifetimeOfAgriculturalInputs", 100);
  averageLifetimeOfAgriculturalInputs.units = "years";
  averageLifetimeOfAgriculturalInputs.before = 2;
  averageLifetimeOfAgriculturalInputs.after = 2;
  averageLifetimeOfAgriculturalInputs.updateFn = function() {
    return clip(this.after, this.before, t, policyYear);
  }

var agriculturalInputsPerHectare = new Aux("agriculturalInputsPerHectare", 101);
  agriculturalInputsPerHectare.units = "dollars per hectare-year";
  agriculturalInputsPerHectare.dependencies = ["agriculturalInputs", "fractionOfInputsAllocatedToLandMaintenance"];
  agriculturalInputsPerHectare.updateFn = function() {
    return agriculturalInputs.k * (1 - fractionOfInputsAllocatedToLandMaintenance.k) / arableLand.k;
  }
  
var landYieldMultiplierFromCapital = new Table("landYieldMultiplierFromCapital", 102, [1, 3, 3.8, 4.4, 4.9, 5.4, 5.7, 6, 6.3, 6.6, 6.9, 7.2, 7.4, 7.6, 7.8, 8, 8.2, 8.4, 8.6, 8.8, 9, 9.2, 9.4, 9.6, 9.8, 10], 0, 1000, 40)
  landYieldMultiplierFromCapital.units = "dimensionless";
  landYieldMultiplierFromCapital.dependencies = ["agriculturalInputsPerHectare"];
  landYieldMultiplierFromCapital.updateFn = function() {
    return agriculturalInputsPerHectare.k;
  }
  
var landYield = new Aux("landYield", 103);
  landYield.units = "kilograms per hectare-year";
  landYield.plotColor = "#185103";
  landYield.plotMin = 0;
  landYield.plotMax = 3000;
  landYield.dependencies = ["landYieldFactor", "landYieldMultiplierFromCapital", "landYieldMultiplierFromAirPollution"]
  landYield.updateFn = function() {
    return landYieldFactor.k *
           landFertility.k *
           landYieldMultiplierFromCapital.k *
           landYieldMultiplierFromAirPollution.k;
  }
  
var landYieldFactor = new Aux("landYieldFactor", 104);
  landYieldFactor.units = "dimensionless";
  landYieldFactor.before = 1;
  landYieldFactor.after = 1;
  landYieldFactor.updateFn = function() {
    return clip(this.after, this.before, t, policyYear);
  }

var landYieldMultiplierFromAirPollution = new Aux("landYieldMultiplierFromAirPollution", 105);
  landYieldMultiplierFromAirPollution.units = "dimensionless";
  landYieldMultiplierFromAirPollution.dependencies = ["landYieldMultiplierFromAirPollutionBefore", "landYieldMultiplierFromAirPollutionAfter"];
  landYieldMultiplierFromAirPollution.updateFn = function() {
    return clip(landYieldMultiplierFromAirPollutionAfter.k, landYieldMultiplierFromAirPollutionBefore.k, t, policyYear);
  }
  
var landYieldMultiplierFromAirPollutionBefore = new Table("landYieldMultiplierFromAirPollutionBefore", 106, [1, 1, 0.7, 0.4], 0, 30, 10)
  landYieldMultiplierFromAirPollutionBefore.units = "dimensionless";
  landYieldMultiplierFromAirPollutionBefore.dependencies = ["industrialOutput"];
  landYieldMultiplierFromAirPollutionBefore.updateFn = function() {
    return industrialOutput.k / industrialOutput.valueIn1970;
  }
  
var landYieldMultiplierFromAirPollutionAfter = new Table("landYieldMultiplierFromAirPollutionAfter", 107, [1, 1, 0.7, 0.4], 0, 30, 10)
  landYieldMultiplierFromAirPollutionAfter.units = "dimensionless";
  landYieldMultiplierFromAirPollutionAfter.dependencies = ["industrialOutput"];
  landYieldMultiplierFromAirPollutionAfter.updateFn = function() {
    return industrialOutput.k / industrialOutput.valueIn1970;
  }
  

// Loops 1 and 2: The Investment Allocation Decision

var fractionOfInputsAllocatedToLandDevelopment = new Table("fractionOfInputsAllocatedToLandDevelopment", 108, [0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1], 0, 2, 0.25);
  fractionOfInputsAllocatedToLandDevelopment.units = "dimensionless";
  fractionOfInputsAllocatedToLandDevelopment.dependencies = ["marginalProductivityOfLandDevelopment", "marginalProductivityOfAgriculturalInputs"];
  fractionOfInputsAllocatedToLandDevelopment.updateFn = function() {
    return marginalProductivityOfLandDevelopment.k / marginalProductivityOfAgriculturalInputs.k;
  }

var marginalProductivityOfLandDevelopment = new Aux("marginalProductivityOfLandDevelopment", 109);
  marginalProductivityOfLandDevelopment.units = "kilograms per dollar";
  marginalProductivityOfLandDevelopment.socialDiscount = 0.07; 
  marginalProductivityOfLandDevelopment.dependencies = ["landYield", "developmentCostPerHectare"]
  marginalProductivityOfLandDevelopment.updateFn = function() {
    return landYield.k / (developmentCostPerHectare.k * marginalProductivityOfLandDevelopment.socialDiscount);
  }
  
var marginalProductivityOfAgriculturalInputs = new Aux("marginalProductivityOfAgriculturalInputs", 110);
  marginalProductivityOfAgriculturalInputs.units = "kilograms per dollar";
  marginalProductivityOfAgriculturalInputs.dependencies = ["averageLifetimeOfAgriculturalInputs", "landYield", "marginalLandYieldMultiplierFromCapital", "landYieldMultiplierFromCapital"]
  marginalProductivityOfAgriculturalInputs.updateFn = function() {
    return averageLifetimeOfAgriculturalInputsK * landYield.k * (marginalLandYieldMultiplierFromCapital.k / landYieldMultiplierFromCapital.k);
  }
  
var marginalLandYieldMultiplierFromCapital = new Table("marginalLandYieldMultiplierFromCapital", 111, [0.075, 0.03, 0.015, 0.011, 0.009, 0.008, 0.007, 0.006, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005], 0, 600, 40)
  marginalLandYieldMultiplierFromCapital.units = "hectares per dollar";
  marginalLandYieldMultiplierFromCapital.dependencies = ["agriculturalInputsPerHectare"];
  marginalLandYieldMultiplierFromCapital.updateFn = function() {
    return agriculturalInputsPerHectare.k;
  }
  

// Loop 3: Land Erosion and Urban-Industrial Use


var averageLifeOfLand = new Aux("averageLifeOfLand", 112);
  averageLifeOfLand.units = "years";
  averageLifeOfLand.normal = 6000;     // years
  averageLifeOfLand.dependencies = ["landLifeMultiplierFromYield"];
  averageLifeOfLand.updateFn = function() {
    return averageLifeOfLand.normal * landLifeMultiplierFromYield.k;
  }
  
var landLifeMultiplierFromYield = new Aux("landLifeMultiplierFromYield", 113);
  landLifeMultiplierFromYield.units = "dimensionless";
  landLifeMultiplierFromYield.dependencies = ["landLifeMultiplierFromYieldBefore", "landLifeMultiplierFromYieldAfter"];
  landLifeMultiplierFromYield.updateFn = function() {
    return clip(landLifeMultiplierFromYieldAfter.k, landLifeMultiplierFromYieldBefore.k, t, policyYear);
  }
  
var inherentLandFertilityK = 600;   // kilograms per hectare-year, used in eqns 114, 115 and 124

var landLifeMultiplierFromYieldBefore = new Table("landLifeMultiplierFromYieldBefore", 114, [1.2, 1, 0.63, 0.36, 0.16, 0.055, 0.04, 0.025, 0.015, 0.01], 0, 9, 1)
  landLifeMultiplierFromYieldBefore.units = "dimensionless";
  landLifeMultiplierFromYieldBefore.dependencies = ["landYield"];
  landLifeMultiplierFromYieldBefore.updateFn = function() {
    return landYield.k / inherentLandFertilityK;
  }
  
var landLifeMultiplierFromYieldAfter = new Table("landLifeMultiplierFromYieldAfter", 115, [1.2, 1, 0.63, 0.36, 0.16, 0.055, 0.04, 0.025, 0.015, 0.01], 0, 9, 1)
  landLifeMultiplierFromYieldAfter.units = "dimensionless";
  landLifeMultiplierFromYieldAfter.dependencies = ["landYield"];
  landLifeMultiplierFromYieldAfter.updateFn = function() {
    return landYield.k / inherentLandFertilityK;
  }
  
var landErosionRate = new Rate("landErosionRate", 116);
  landErosionRate.units = "hectares per year";
  landErosionRate.updateFn = function() {
    return arableLand.k / averageLifeOfLand.k;
  }
  
// 2016-08-09: Neil S. Grant reported an error in the table of values
// for urbanIndustrialLandPerCapita. The third element of the array
// should be 0.015, not 0.15. Corrected.
  
var urbanIndustrialLandPerCapita = new Table("urbanIndustrialLandPerCapita", 117, [0.005, 0.008, 0.015, 0.025, 0.04, 0.055, 0.07, 0.08, 0.09], 0, 1600, 200);
  urbanIndustrialLandPerCapita.units = "hectares per person";
  urbanIndustrialLandPerCapita.dependencies = ["industrialOutputPerCapita"];
  urbanIndustrialLandPerCapita.updateFn = function() {
    return industrialOutputPerCapita.k;
  }

var urbanIndustrialLandRequired = new Aux("urbanIndustrialLandRequired", 118);
  urbanIndustrialLandRequired.units = "hectares";
  urbanIndustrialLandRequired.dependencies = ["urbanIndustrialLandPerCapita", "population"];
  urbanIndustrialLandRequired.updateFn = function() {
    return urbanIndustrialLandPerCapita.k * population.k;
  }
  
var landRemovalForUrbanIndustrialUse = new Rate("landRemovalForUrbanIndustrialUse", 119);
  landRemovalForUrbanIndustrialUse.units = "hectares per year";
  landRemovalForUrbanIndustrialUse.developmentTime = 10;   // years
  landRemovalForUrbanIndustrialUse.updateFn = function() {
    return Math.max(0, (urbanIndustrialLandRequired.k - urbanIndustrialLand.k) / landRemovalForUrbanIndustrialUse.developmentTime);
  }
  
var urbanIndustrialLand = new Level("urbanIndustrialLand", 120, 8.2e6);
  urbanIndustrialLand.units = "hectares";
  urbanIndustrialLand.updateFn = function() {
    return urbanIndustrialLand.j + dt * landRemovalForUrbanIndustrialUse.j;
  }
  

// Loop 4: Land fertility degradation

var landFertility = new Level("landFertility", 121, 600);
  landFertility.units = "kilograms per hectare-year";
  landFertility.updateFn = function() {
    return landFertility.j + dt * (landFertilityRegeneration.j - landFertilityDegradation.j);
  }
  
var landFertilityDegradationRate = new Table("landFertilityDegradationRate", 122, [0, 0.1, 0.3, 0.5], 0, 30, 10);
  landFertilityDegradationRate.units = "inverse years";
  landFertilityDegradationRate.dependencies = ["indexOfPersistentPollution"];
  landFertilityDegradationRate.updateFn = function() {
    return indexOfPersistentPollution.k;
  }

var landFertilityDegradation = new Rate("landFertilityDegradation", 123);
  landFertilityDegradation.units = "kilograms per hectare-year-year";
  landFertilityDegradation.updateFn = function() {
    return landFertility.k * landFertilityDegradationRate.k;
  }
  


// Loop 5: Land fertility regeneration


var landFertilityRegeneration = new Rate("landFertilityRegeneration", 124);
  landFertilityRegeneration.units = "kilograms per hectare-year-year";
  landFertilityRegeneration.updateFn = function() {
    return (inherentLandFertilityK - landFertility.k) / landFertilityRegenerationTime.k;
  }
  
var landFertilityRegenerationTime = new Table("landFertilityRegenerationTime", 125, [20, 13, 8, 4, 2, 2], 0, 0.1, 0.02);
  landFertilityRegenerationTime.units = "years";
  landFertilityRegenerationTime.dependencies = ["fractionOfInputsAllocatedToLandMaintenance"];
  landFertilityRegenerationTime.updateFn = function() {
    return fractionOfInputsAllocatedToLandMaintenance.k;
  }


// Loop 6: Discontinuing land maintenance

var fractionOfInputsAllocatedToLandMaintenance = new Table("fractionOfInputsAllocatedToLandMaintenance", 126, [0, 0.04, 0.07, 0.09, 0.10], 0, 4, 1);
  fractionOfInputsAllocatedToLandMaintenance.units = "dimensionless";
  fractionOfInputsAllocatedToLandMaintenance.dependencies = ["perceivedFoodRatio"];
  fractionOfInputsAllocatedToLandMaintenance.updateFn = function() {
    return perceivedFoodRatio.k;
  }

var foodRatio = new Aux("foodRatio", 127);
  foodRatio.units = "dimensionless";
  foodRatio.dependencies = ["foodPerCapita"];
  foodRatio.updateFn = function() {
    return foodPerCapita.k / subsistenceFoodPerCapitaK;
  }
  
var foodShortagePerceptionDelayK = 2;  // years, used in eqn 128
  
var perceivedFoodRatio = new Smooth("perceivedFoodRatio", 128, foodShortagePerceptionDelayK);
  perceivedFoodRatio.units = "dimensionless";
  perceivedFoodRatio.dependencies = [];   // "foodRatio" removed to break cycle
  perceivedFoodRatio.initFn = function() { return foodRatio; }
  perceivedFoodRatio.initVal = 1.0;


/*
var perceivedFoodRatio = new Smooth("perceivedFoodRatio", 128, foodShortagePerceptionDelayK);
  perceivedFoodRatio.units = "dimensionless";
  perceivedFoodRatio.dependencies = [];   // "foodRatio" removed to break cycle
  perceivedFoodRatio.initFn = function() { return foodRatio; }
  perceivedFoodRatio.init = function() {
    perceivedFoodRatio.theInput = perceivedFoodRatio.initFn;
    perceivedFoodRatio.j = perceivedFoodRatio.k = 1.0;    
  }
  perceivedFoodRatio.update = function() {
    if (perceivedFoodRatio.firstCall) {
      perceivedFoodRatio.firstCall = false;
      return perceivedFoodRatio.k;
    }
    else {
      perceivedFoodRatio.k = perceivedFoodRatio.j + dt * (perceivedFoodRatio.theInput.j - perceivedFoodRatio.j) / perceivedFoodRatio.del;
      return perceivedFoodRatio.k;
    }
  }
*/
  
  
// NONRENEWABLE RESOURCE SECTOR


var nonrenewableResourcesInitialK = 1.0e12;  // resource units, used in eqns 129 and 133

var nonrenewableResources = new Level("nonrenewableResources", 129, nonrenewableResourcesInitialK);
  nonrenewableResources.units = "resource units";
  nonrenewableResources.updateFn = function() {
    return nonrenewableResources.j + dt * (-nonrenewableResourceUsageRate.j);
  }
  
var nonrenewableResourceUsageRate = new Rate("nonrenewableResourceUsageRate", 130);
  nonrenewableResourceUsageRate.units = "resource units per year";
  nonrenewableResourceUsageRate.updateFn = function() {
    return population.k * perCapitaResourceUsageMultiplier.k * nonrenewableResourceUsageFactor.k;
  }
  
var nonrenewableResourceUsageFactor = new Aux("nonrenewableResourceUsageFactor", 131);
  nonrenewableResourceUsageFactor.units = "dimensionless";
  nonrenewableResourceUsageFactor.before = 1;
  nonrenewableResourceUsageFactor.after = 1;
  nonrenewableResourceUsageFactor.updateFn = function() {
    return clip(this.after, this.before, t, policyYear);
  }

var perCapitaResourceUsageMultiplier = new Table("perCapitaResourceUsageMultiplier", 132, [0, 0.85, 2.6, 4.4, 5.4, 6.2, 6.8, 7, 7], 0, 1600, 200);
  perCapitaResourceUsageMultiplier.units = "resource units per person-year";
  perCapitaResourceUsageMultiplier.dependencies = ["industrialOutputPerCapita"];
  perCapitaResourceUsageMultiplier.updateFn = function() {
    return industrialOutputPerCapita.k;
  }

var nonrenewableResourceFractionRemaining = new Aux("nonrenewableResourceFractionRemaining", 133);
  nonrenewableResourceFractionRemaining.units = "dimensionless";
  nonrenewableResourceFractionRemaining.plotColor = "#b0875e";
  nonrenewableResourceFractionRemaining.plotMin = 0.0;
  nonrenewableResourceFractionRemaining.plotMax = 1.0;
  nonrenewableResourceFractionRemaining.updateFn = function() {
    return nonrenewableResources.k / nonrenewableResourcesInitialK;
  }
  
var fractionOfCapitalAllocatedToObtainingResources = new Aux("fractionOfCapitalAllocatedToObtainingResources", 134);
  fractionOfCapitalAllocatedToObtainingResources.units = "dimensionless";
  fractionOfCapitalAllocatedToObtainingResources.dependencies = ["fractionOfCapitalAllocatedToObtainingResourcesBefore", "fractionOfCapitalAllocatedToObtainingResourcesAfter"];
  fractionOfCapitalAllocatedToObtainingResources.updateFn = function() {
    return clip(fractionOfCapitalAllocatedToObtainingResourcesAfter.k, fractionOfCapitalAllocatedToObtainingResourcesBefore.k, t, policyYear);
  }
  
var fractionOfCapitalAllocatedToObtainingResourcesBefore = new Table("fractionOfCapitalAllocatedToObtainingResourcesBefore", 135, [1, 0.9, 0.7, 0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05], 0, 1, 0.1);
  fractionOfCapitalAllocatedToObtainingResourcesBefore.units = "dimensionless";
  fractionOfCapitalAllocatedToObtainingResourcesBefore.dependencies = ["nonrenewableResourceFractionRemaining"];
  fractionOfCapitalAllocatedToObtainingResourcesBefore.updateFn = function() {
    return nonrenewableResourceFractionRemaining.k;
  }
  
var fractionOfCapitalAllocatedToObtainingResourcesAfter = new Table("fractionOfCapitalAllocatedToObtainingResourcesAfter", 136, [1, 0.9, 0.7, 0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05], 0, 1, 0.1);
  fractionOfCapitalAllocatedToObtainingResourcesAfter.units = "dimensionless";
  fractionOfCapitalAllocatedToObtainingResourcesAfter.dependencies = ["nonrenewableResourceFractionRemaining"];
  fractionOfCapitalAllocatedToObtainingResourcesAfter.updateFn = function() {
    return nonrenewableResourceFractionRemaining.k;
  }
  

// PERSISTENT POLLUTION SECTOR


var persistentPollutionGenerationRate = new Rate("persistentPollutionGenerationRate", 137);
  persistentPollutionGenerationRate.units = "pollution units per year";
  persistentPollutionGenerationRate.updateFn = function() {
    return (persistentPollutionGeneratedByIndustrialOutput.k + persistentPollutionGeneratedByAgriculturalOutput.k) * persistentPollutionGenerationFactor.k;
  }
  
var persistentPollutionGenerationFactor = new Aux("persistentPollutionGenerationFactor", 138);
  persistentPollutionGenerationFactor.units = "dimensionless";
  persistentPollutionGenerationFactor.before = 1;
  persistentPollutionGenerationFactor.after = 1;
  persistentPollutionGenerationFactor.updateFn = function() {
    return clip(this.after, this.before, t, policyYear);
  }

var persistentPollutionGeneratedByIndustrialOutput = new Aux("persistentPollutionGeneratedByIndustrialOutput", 139);
  persistentPollutionGeneratedByIndustrialOutput.units = "pollution units per year";
  persistentPollutionGeneratedByIndustrialOutput.fractionOfResourcesAsPersistentMaterial = 0.02;  // dimensionless
  persistentPollutionGeneratedByIndustrialOutput.industrialMaterialsEmissionFactor = 0.1;  // dimensionless
  persistentPollutionGeneratedByIndustrialOutput.industrialMaterialsToxicityIndex = 10;  // pollution units per resource unit  
  persistentPollutionGeneratedByIndustrialOutput.dependencies = ["perCapitaResourceUsageMultiplier", "population"];
  persistentPollutionGeneratedByIndustrialOutput.updateFn = function() {
    return perCapitaResourceUsageMultiplier.k * population.k * persistentPollutionGeneratedByIndustrialOutput.fractionOfResourcesAsPersistentMaterial * persistentPollutionGeneratedByIndustrialOutput.industrialMaterialsEmissionFactor * persistentPollutionGeneratedByIndustrialOutput.industrialMaterialsToxicityIndex;
  }
  
var persistentPollutionGeneratedByAgriculturalOutput = new Aux("persistentPollutionGeneratedByAgriculturalOutput", 140);
  persistentPollutionGeneratedByAgriculturalOutput.units = "pollution units per year";
  persistentPollutionGeneratedByAgriculturalOutput.fractionOfInputsAsPersistentMaterial = 0.001;  // dimensionless
  persistentPollutionGeneratedByAgriculturalOutput.agriculturalMaterialsToxicityIndex = 1;  // pollution units per dollar  
  persistentPollutionGeneratedByAgriculturalOutput.dependencies = ["agriculturalInputsPerHectare"];
  persistentPollutionGeneratedByAgriculturalOutput.updateFn = function() {
    return agriculturalInputsPerHectare.k * arableLand.k * persistentPollutionGeneratedByAgriculturalOutput.fractionOfInputsAsPersistentMaterial * persistentPollutionGeneratedByAgriculturalOutput.agriculturalMaterialsToxicityIndex;
  }
  
var persistentPollutionTransmissionDelayK = 20; // years, used in eqn 141  

var persistenPollutionAppearanceRate = new Delay3("persistenPollutionAppearanceRate", 141, persistentPollutionTransmissionDelayK);
  persistenPollutionAppearanceRate.units = "pollution units per year";
  persistenPollutionAppearanceRate.initFn = function() {return persistentPollutionGenerationRate; }
  persistenPollutionAppearanceRate.qType = "Rate";
  rateArray.push(auxArray.pop());   // put this among the Rates, not the Auxes
  
var persistentPollution = new Level("persistentPollution", 142, 2.5e7);
  persistentPollution.units = "pollution units";
  persistentPollution.updateFn = function() {
    return persistentPollution.j + dt * (persistenPollutionAppearanceRate.j - persistenPollutionAssimilationRate.j);
  }
  
var indexOfPersistentPollution = new Aux("indexOfPersistentPollution", 143);
  indexOfPersistentPollution.units = "dimensionless";
  indexOfPersistentPollution.pollutionValueIn1970 = 1.36e8; // pollution units, used in eqn 143
  indexOfPersistentPollution.plotColor = "#a25563";
  indexOfPersistentPollution.plotMin = 0;
  indexOfPersistentPollution.plotMax = 32;
  indexOfPersistentPollution.updateFn = function() {
    return persistentPollution.k / indexOfPersistentPollution.pollutionValueIn1970;
  }
  
var persistenPollutionAssimilationRate = new Rate("persistenPollutionAssimilationRate", 144);
  persistenPollutionAssimilationRate.units = "pollution units per year";
  persistenPollutionAssimilationRate.updateFn = function() {
    return persistentPollution.k / (assimilationHalfLife.k * 1.4);
  }
  
var assimilationHalfLifeMultiplier = new Table("assimilationHalfLifeMultiplier", 145, [1, 11, 21, 31, 41], 1, 1001, 250);
  assimilationHalfLifeMultiplier.units = "dimensionless";
  assimilationHalfLifeMultiplier.dependencies = ["indexOfPersistentPollution"];
  assimilationHalfLifeMultiplier.updateFn = function() {
    return indexOfPersistentPollution.k;
  }

var assimilationHalfLife = new Aux("assimilationHalfLife", 146);
  assimilationHalfLife.units = "years";
  assimilationHalfLife.valueIn1970 = 1.5; // years
  assimilationHalfLife.dependencies = ["assimilationHalfLifeMultiplier"];
  assimilationHalfLife.updateFn = function() {
    return assimilationHalfLifeMultiplier.k * assimilationHalfLife.valueIn1970;
  }
  
// SUPPLEMENTARY EQUATIONS

var fractionOfOutputInAgriculture = new Aux("fractionOfOutputInAgriculture", 147);
  fractionOfOutputInAgriculture.units = "dimensionless";
  fractionOfOutputInAgriculture.dependencies = ["food", "serviceOutput", "industrialOutput"]
  fractionOfOutputInAgriculture.updateFn = function() {
    return 0.22 * food.k / ((0.22 * food.k) + serviceOutput.k + industrialOutput.k);
  }

var fractionOfOutputInIndustry = new Aux("fractionOfOutputInIndustry", 148);
  fractionOfOutputInIndustry.units = "dimensionless";
  fractionOfOutputInIndustry.dependencies = ["food", "serviceOutput", "industrialOutput"]
  fractionOfOutputInIndustry.updateFn = function() {
    return industrialOutput.k / (0.22 * food.k + serviceOutput.k + industrialOutput.k);
  }

var fractionOfOutputInServices = new Aux("fractionOfOutputInServices", 149);
  fractionOfOutputInServices.units = "dimensionless";
  fractionOfOutputInServices.dependencies = ["food", "serviceOutput", "industrialOutput"]
  fractionOfOutputInServices.updateFn = function() {
    return serviceOutput.k / (0.22 * food.k + serviceOutput.k + industrialOutput.k);
  }




// ENTRY POINT: called by body.onload

var setUpModel = function() {
  setUpGraph();
  setUpControls();
  setDefaults();
}






// GRAPHICS


// some basic dimensions

var cvWidth = 800;
var cvHeight = 450;
var gLeft = 50;
var gRight = cvWidth - 50;
var gTop = 25;
var gBottom = cvHeight - 50;


// RGB colors associated with the polttable variables


var scaleX = function(x, xMin, xMax) {
  var sx = (x - xMin) / (xMax - xMin);
  var px = gLeft + sx * (gRight - gLeft);
  return px;
}

var scaleY = function(y, yMin, yMax) {
  var sy = (y - yMin) / (yMax - yMin);
  var py = gTop + (1 - sy) * (gBottom - gTop)
  return py;
}

// There's nothing to write on!
var setUpGraph = function() {
//  var cv = document.getElementById("cv");
//  cv.width = cv.width;
//  var cvx = cv.getContext("2d");
//
//  // draw horizontal gridlines
//  
//  cvx.lineWidth = 1;
//  cvx.strokeStyle = "#fff"
//  for (var y = 0 ; y <= 5 ; y++) {
//    cvx.moveTo(scaleX(0, 0, 1), scaleY(y, 0, 5));
//    cvx.lineTo(scaleX(1, 0, 1), scaleY(y, 0, 5));
//    cvx.stroke();
//  }
//  
//  // draw vertical gridlines
//  
//  cvx.lineWidth = 1;
//  cvx.strokeStyle = "#fff"
//  for (var x = startTime ; x <= stopTime ; x += 50) {
//    cvx.moveTo(scaleX(x, startTime, stopTime), scaleY(0, 0, 1));
//    cvx.lineTo(scaleX(x, startTime, stopTime), scaleY(1, 0, 1));
//    cvx.stroke();
//  }
//  
//  // place labels for time axis
//  
//  cvx.font = "1.0em 'Helvetica Neue', Helvetica, Verdana, sans-serif";
//  cvx.textAlign = "center";
//  cvx.fillStyle = "#000";
//  var textY = gBottom + 20;
//  for (var textX = startTime ; textX <= stopTime ; textX += 50) {
//    cvx.fillText(textX.toString(), scaleX(textX, startTime, stopTime), textY);
//  }
//  cvx.fillText("year", scaleX(1, 0, 2), gBottom + 40);
//  
}



var plotLine = function(data, yMin, yMax, color) {
  var cvx = document.getElementById("cv").getContext("2d");
  cvx.strokeStyle = color;
  cvx.beginPath();
  var leftPoint = data.shift();
  cvx.moveTo(scaleX(leftPoint.x, startTime, stopTime), scaleY(leftPoint.y, yMin, yMax));
  for (var i = 0 ; i < data.length ; i++) {
    var p = data[i];
    cvx.lineTo(scaleX(p.x, startTime, stopTime), scaleY(p.y, yMin, yMax));
  }
  cvx.stroke();
  cvx.closePath();
}


var testPlotData = [ {x: 1900, y: 1.6e9}, {x: 1910, y: 1.7e9}, {x: 1920, y: 1.9e9}, {x: 2100, y: 1.1e9} ];







// CONTROLS

// array of plottable variables

var plottable = ["agriculturalInputs", "agriculturalInputsPerHectare", "assimilationHalfLife", "averageLifeOfLand", "capitalUtilizationFraction", "effectiveHealthServicesPerCapita", "familyIncomeExpectation", "familyResponseToSocialNorm", "fecundityMultiplier", "fertilityControlAllocationPerCapita", "food", "fractionOfCapitalAllocatedToObtainingResources", "fractionOfIndustrialOutputAllocatedToAgriculture", "fractionOfIndustrialOutputAllocatedToIndustry", "fractionOfIndustrialOutputAllocatedToServices", "fractionOfInputsAllocatedToLandDevelopment", "fractionOfInputsAllocatedToLandMaintenance", "fractionOfOutputInAgriculture", "fractionOfOutputInIndustry", "fractionOfOutputInServices", "fractionOfPopulationUrban", "fractionOfServicesAllocatedToFertilityControl", "healthServicesAllocationsPerCapita", "industrialCapital", "industrialCapitalDepreciationRate", "industrialCapitalInvestmentRate", "industrialCapitalOutputRatio", "industrialOutput", "jobs", "jobsPerHectare", "jobsPerIndustrialCapitalUnit", "jobsPerServiceCapitalUnit", "laborForce", "laborUtilizationFraction", "landDevelopmentRate", "landErosionRate", "landFertility", "landFertilityDegradation", "landFertilityDegradationRate", "landFertilityRegeneration", "landFractionCultivated", "landLifeMultiplierFromYield", "landRemovalForUrbanIndustrialUse", "lifetimeMultiplierFromCrowding", "lifetimeMultiplierFromFood", "lifetimeMultiplierFromHealthServices", "lifetimeMultiplierFromPollution", "mortality0To14", "mortality15To44", "mortality45To64", "mortality65AndOver", "needForFertilityControl", "nonrenewableResourceUsageRate", "perCapitaResourceUsageMultiplier", "perceivedFoodRatio", "perceivedLifeExpectancy", "persistenPollutionAppearanceRate", "persistenPollutionAssimilationRate",  "persistentPollutionGenerationRate", "potentialJobsInAgriculturalSector", "potentialJobsInIndustrialSector", "potentialJobsInServiceSector", "potentiallyArableLand", "serviceCapital", "serviceCapitalDepreciationRate", "serviceCapitalInvestmentRate", "serviceCapitalOutputRatio", "serviceOutput", "socialFamilySizeNorm", "totalAgriculturalInvestment", "totalFertility", "urbanIndustrialLand"]




// add variables to the pop-up menu

/*
var populateMenu = function() {
  var menu = document.getElementById("menuOfVars")
  for (var i in plottable) {
    var iOption = new Option();
    iOption.text = plottable[i];
    menu.options[menu.length] = iOption;
  }
}
*/




var setUpControls = function() {
  pollCheckBoxes();
  changeDuration();
  changeDt();
  changeResources();
  changeConsumption();
}


var changeDuration = function() {
  var sliderInput = parseInt(document.getElementById("duration-slider").value);
  var sliderReadOut = document.getElementById("duration-readout");
  sliderReadOut.innerHTML = sliderInput.toString();
  stopTime = startTime + sliderInput;
  resetModel();
  setUpGraph();
}

var changeDt = function() {
  var sliderInput = parseInt(document.getElementById("dt-slider").value);
  var sliderReadOut = document.getElementById("dt-readout");
  var newDt = Math.pow(2, sliderInput);
  sliderReadOut.innerHTML = newDt.toString();
  dt = newDt;
  resetModel();
}

var changeResources = function() {
  var sliderInput = parseInt(document.getElementById("resource-slider").value);
  var sliderReadOut = document.getElementById("resource-readout");
  var newResources = Math.pow(2, sliderInput);
  sliderReadOut.innerHTML = newResources.toString();
  nonrenewableResources.initVal = newResources * 1.0e12;
  nonrenewableResourcesInitialK = newResources * 1.0e12;
  resetModel();
}

var changeConsumption = function() {
  var sliderInput = parseFloat(document.getElementById("consumption-slider").value);
  var sliderReadOut = document.getElementById("consumption-readout");
  sliderReadOut.innerHTML = sliderInput.toFixed(2);
  fractionOfIndustrialOutputAllocatedToConsumptionConstant.before = sliderInput;
  fractionOfIndustrialOutputAllocatedToConsumptionConstant.after = sliderInput;
  resetModel();
}

/*
var changeMenuVar = function() {
  var menu = document.getElementById("menuOfVars");
  var menuCheckBox = document.getElementById("select-var-ck");
  var selection = menu.options[menu.selectedIndex].text;
  menuCheckBox.name = selection;

}
*/


var disableControls = function() {
  var ctrls = document.getElementsByTagName("input");
  for (var c = 0 ; c < ctrls.length ; c++) {
    ctrls[c].setAttribute("disabled", "disabled");
  }
  var btns = document.getElementsByTagName("button");
  for (var b = 0 ; b < btns.length ; b++) {
    if (btns[b].getAttribute("id") != "run") {
      btns[b].setAttribute("disabled", "disabled");
    }
  }
}

var enableControls = function() {
  var ctrls = document.getElementsByTagName("input");
  for (var c = 0 ; c < ctrls.length ; c++) {
    ctrls[c].removeAttribute("disabled");
  }
  var btns = document.getElementsByTagName("button");
  for (var b = 0 ; b < btns.length ; b++) {
    if (btns[b].getAttribute("id") != "run") {
      btns[b].removeAttribute("disabled");
    }
  }
}

var setStopButton = function() {
  var btn = document.getElementById("run");
    btn.setAttribute("onclick", "stopModel()");
    btn.innerHTML = "Stop";
}

var setRunButton = function() {
  var btn = document.getElementById("run");
    btn.setAttribute("onclick", "runModel()");
    btn.innerHTML = "Run";
}



var pollCheckBoxes = function() {
  var ckx = document.getElementsByClassName("checkbox-line");
  for (var i = 0 ; i < ckx.length ; i++) {
    var theInput = ckx[i].getElementsByTagName("input")[0];
    var theEqn = eval(theInput.getAttribute("name"));
    var theSample = ckx[i].getElementsByClassName("color-sample")[0];
    var theHue = theEqn.plotColor;
    if (theInput.checked == true) {
      theSample.style.backgroundColor = theHue;
      theEqn.plotThisVar = true;
    }
    else {
      theSample.style.backgroundColor = "transparent";
      theEqn.plotThisVar = false;    
    }    
  }
}

var setDefaults = function() {
  var plotVars = [ "population-ck", "resources-ck", "food-ck", "industry-ck", "pollution-ck", "life-expect-ck" ];
  var ckx = document.getElementsByClassName("checkbox-line");
  for (var i = 0 ; i < ckx.length ; i++) {
    var theInput = ckx[i].getElementsByTagName("input")[0];
    theInput.checked = false;
  }
  for (var id in plotVars) {
    var ckBox = document.getElementById(plotVars[id]);
    ckBox.checked = true;
  }
  pollCheckBoxes();
  var duration = document.getElementById("duration-slider");
  duration.value = 200;
  changeDuration();
  var dtx = document.getElementById("dt-slider");
  dtx.value = -1;
  changeDt();
  var res = document.getElementById("resource-slider");
  res.value = 0;
  changeResources();
  var cons = document.getElementById("consumption-slider");
  cons.value = 0.43;
  changeConsumption();
}




// DEBUG LOGGING

var logData = function() {
  var vals = [t, foodRatio.k, perceivedFoodRatio.k];
  var valStr = vals.join("  ");
  console.log(valStr);
}

