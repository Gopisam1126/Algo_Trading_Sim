let ws = null;
let messageCount = 0;
let connectionStartTime = null;
let connectionTimer = null;
let priceHistory = [];
const maxPriceHistory = 100;

// Technical indicators configuration
let indicatorsConfig = {
  sma_periods: [20, 50, 200],
  ema_periods: [12, 26, 50],
  macd_fast: 12,
  macd_slow: 26,
  macd_signal: 9,
  rsi_period: 14,
  adx_period: 14,
  psar_acceleration: 0.02,
  psar_maximum: 0.2,
  ichimoku_conversion: 9,
  ichimoku_base: 26,
  ichimoku_span_b: 52,
  ichimoku_displacement: 26,
  supertrend_period: 10,
  supertrend_multiplier: 3.0,
  stochastic_k_period: 14,
  stochastic_d_period: 3,
  cci_period: 20,
  roc_period: 14,
  momentum_period: 10,
  momentum_threshold_multiplier: 0.02,
  atr_period: 14,
  fibonacci_period: 20,
  fibonacci_levels: [0.236, 0.382, 0.5, 0.618, 0.786],
};

// Technical indicators data storage
let indicatorsData = {
  moving_averages: {},
  macd: {},
  rsi: {},
  adx: {},
  parabolic_sar: {},
  ichimoku: {},
  supertrend: {},
  stochastic: {},
  cci: {},
  roc: {},
  momentum: {},
  atr: {},
  fibonacci: {},
};

function updateIndicatorsConfig() {
  try {
    // Get configuration from form inputs
    const smaPeriodsStr = document.getElementById("smaPeriods").value;
    const emaPeriodsStr = document.getElementById("emaPeriods").value;
    const fibonacciLevelsStr = document.getElementById("fibonacciLevels").value;

    indicatorsConfig = {
      sma_periods: smaPeriodsStr
        .split(",")
        .map((p) => parseInt(p.trim()))
        .filter((p) => !isNaN(p)),
      ema_periods: emaPeriodsStr
        .split(",")
        .map((p) => parseInt(p.trim()))
        .filter((p) => !isNaN(p)),
      macd_fast: parseInt(document.getElementById("macdFast").value) || 12,
      macd_slow: parseInt(document.getElementById("macdSlow").value) || 26,
      macd_signal: parseInt(document.getElementById("macdSignal").value) || 9,
      rsi_period: parseInt(document.getElementById("rsiPeriod").value) || 14,
      adx_period: parseInt(document.getElementById("adxPeriod").value) || 14,
      psar_acceleration:
        parseFloat(document.getElementById("psarAcceleration").value) || 0.02,
      psar_maximum:
        parseFloat(document.getElementById("psarMaximum").value) || 0.2,
      ichimoku_conversion:
        parseInt(document.getElementById("ichimokuConversion").value) || 9,
      ichimoku_base:
        parseInt(document.getElementById("ichimokuBase").value) || 26,
      ichimoku_span_b:
        parseInt(document.getElementById("ichimokuSpanB").value) || 52,
      ichimoku_displacement: 26,
      supertrend_period:
        parseInt(document.getElementById("supertrendPeriod").value) || 10,
      supertrend_multiplier:
        parseFloat(document.getElementById("supertrendMultiplier").value) ||
        3.0,
      stochastic_k_period:
        parseInt(document.getElementById("stochasticKPeriod").value) || 14,
      stochastic_d_period:
        parseInt(document.getElementById("stochasticDPeriod").value) || 3,
      cci_period: parseInt(document.getElementById("cciPeriod").value) || 20,
      roc_period: parseInt(document.getElementById("rocPeriod").value) || 14,
      momentum_period:
        parseInt(document.getElementById("momentumPeriod").value) || 10,
      momentum_threshold_multiplier:
        parseFloat(document.getElementById("momentumThreshold").value) || 0.02,
      atr_period: parseInt(document.getElementById("atrPeriod")?.value) || 14,
      fibonacci_period:
        parseInt(document.getElementById("fibonacciPeriod").value) || 20,
      fibonacci_levels: fibonacciLevelsStr
        .split(",")
        .map((l) => parseFloat(l.trim()))
        .filter((l) => !isNaN(l)),
    };

    addLog("📊 Technical indicators configuration updated", "info");

    // Send configuration to API
    fetch("/api/indicators/config", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(indicatorsConfig),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        addLog("✅ Configuration sent to API successfully", "success");
      })
      .catch((error) => {
        addLog(
          "❌ Failed to send configuration to API: " + error.message,
          "error"
        );
      });
  } catch (error) {
    addLog("❌ Error updating indicators config: " + error.message, "error");
  }
}

function resetConfig() {
  try {
    document.getElementById("smaPeriods").value = "20,50,200";
    document.getElementById("emaPeriods").value = "12,26,50";
    document.getElementById("macdFast").value = "12";
    document.getElementById("macdSlow").value = "26";
    document.getElementById("macdSignal").value = "9";
    document.getElementById("rsiPeriod").value = "14";
    document.getElementById("adxPeriod").value = "14";
    document.getElementById("psarAcceleration").value = "0.02";
    document.getElementById("psarMaximum").value = "0.2";
    document.getElementById("ichimokuConversion").value = "9";
    document.getElementById("ichimokuBase").value = "26";
    document.getElementById("ichimokuSpanB").value = "52";
    document.getElementById("supertrendPeriod").value = "10";
    document.getElementById("supertrendMultiplier").value = "3.0";
    document.getElementById("stochasticKPeriod").value = "14.0";
    document.getElementById("stochasticDPeriod").value = "3.0";
    document.getElementById("cciPeriod").value = "20";
    document.getElementById("rocPeriod").value = "14";
    document.getElementById("momentumPeriod").value = "10";
    document.getElementById("momentumThreshold").value = "0.02";
    document.getElementById("atrPeriod").value = "14";
    document.getElementById("fibonacciPeriod").value = "20";
    document.getElementById("fibonacciLevels").value =
      "0.236,0.382,0.5,0.618,0.786";

    addLog("🔄 Configuration reset to defaults", "info");
  } catch (error) {
    addLog("❌ Error resetting config: " + error.message, "error");
  }
}

function saveConfig() {
  updateIndicatorsConfig();
  addLog("💾 Configuration saved", "success");
}

function toggleConfig() {
  try {
    const configPanel = document.getElementById("configPanel");
    configPanel.style.display =
      configPanel.style.display === "none" ? "block" : "none";
  } catch (error) {
    addLog("❌ Error toggling config panel: " + error.message, "error");
  }
}

function fetchTechnicalIndicators() {
  // Check if we're running locally or have CORS configured
  const apiUrl =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
      ? "http://localhost:5000/api/indicators/current"
      : "/api/indicators/current";

  fetch(apiUrl)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      console.log("Received data from API:", data);
      if (data.data && data.data.technical_indicators) {
        console.log("Using data.data.technical_indicators");
        updateIndicatorsDisplay(data.data.technical_indicators);
      } else if (data.technical_indicators) {
        console.log("Using data.technical_indicators");
        updateIndicatorsDisplay(data.technical_indicators);
      } else {
        console.log("No technical indicators found in data");
        showIndicatorsUnavailable();
      }
    })
    .catch((error) => {
      console.error("Error fetching technical indicators:", error);
      showIndicatorsUnavailable();
      addLog(
        "❌ Error fetching technical indicators: " + error.message,
        "error"
      );
    });
}

function showIndicatorsUnavailable() {
  const sections = [
    "movingAveragesData",
    "macdData",
    "rsiData",
    "adxData",
    "psarData",
    "ichimokuData",
    "supertrendData",
    "stochasticData",
    "cciData",
    "momentumData",
    "atrData",
    "fibonacciData",
  ];
  sections.forEach((sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.innerHTML = "<div>API not available</div>";
    }
  });
}

function updateIndicatorsDisplay(indicators) {
  try {
    console.log("Updating indicators display with:", indicators);

    // Update Moving Averages
    let maHtml = "";
    for (const [key, value] of Object.entries(indicators)) {
      if (key.startsWith("sma_") || key.startsWith("ema_")) {
        if (value !== null && value !== undefined) {
          maHtml += `<div><strong>${key.toUpperCase()}:</strong> ${Number(
            value
          ).toFixed(2)}</div>`;
        }
      }
    }
    const maElement = document.getElementById("movingAveragesData");
    if (maElement) {
      maElement.innerHTML = maHtml || "<div>No data available</div>";
    }

    // Update MACD
    if (indicators.macd) {
      const macd = indicators.macd;
      const macdElement = document.getElementById("macdData");
      if (macdElement) {
        macdElement.innerHTML = `
                            <div><strong>MACD Line:</strong> ${
                              macd.macd_line?.toFixed(4) || "N/A"
                            }</div>
                            <div><strong>Signal Line:</strong> ${
                              macd.signal_line?.toFixed(4) || "N/A"
                            }</div>
                            <div><strong>Histogram:</strong> ${
                              macd.histogram?.toFixed(4) || "N/A"
                            }</div>
                            <div><strong>Fast EMA:</strong> ${
                              macd.fast_ema?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>Slow EMA:</strong> ${
                              macd.slow_ema?.toFixed(2) || "N/A"
                            }</div>
                        `;
      }
    }

    // Update RSI
    if (indicators.rsi) {
      const rsi = indicators.rsi;
      const rsiElement = document.getElementById("rsiData");
      if (rsiElement) {
        rsiElement.innerHTML = `
                            <div><strong>RSI:</strong> ${
                              rsi.value?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>Signal:</strong> ${
                              rsi.signal || "N/A"
                            }</div>
                        `;
      }
    }

    // Update ADX
    if (indicators.adx) {
      const adx = indicators.adx;
      const adxElement = document.getElementById("adxData");
      if (adxElement) {
        adxElement.innerHTML = `
                            <div><strong>ADX:</strong> ${
                              adx.adx?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>+DI:</strong> ${
                              adx.plus_di?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>-DI:</strong> ${
                              adx.minus_di?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>DX:</strong> ${
                              adx.dx?.toFixed(2) || "N/A"
                            }</div>
                        `;
      }
    }

    // Update Parabolic SAR
    if (indicators.parabolic_sar) {
      const psar = indicators.parabolic_sar;
      const psarElement = document.getElementById("psarData");
      if (psarElement) {
        psarElement.innerHTML = `
                            <div><strong>SAR:</strong> ${
                              psar.value?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>Signal:</strong> ${
                              psar.signal || "N/A"
                            }</div>
                        `;
      }
    }

    // Update Ichimoku Cloud
    if (indicators.ichimoku) {
      const ichimoku = indicators.ichimoku;
      const ichimokuElement = document.getElementById("ichimokuData");
      if (ichimokuElement) {
        ichimokuElement.innerHTML = `
                            <div><strong>Conversion:</strong> ${
                              ichimoku.conversion_line?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>Base:</strong> ${
                              ichimoku.base_line?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>Span A:</strong> ${
                              ichimoku.span_a?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>Span B:</strong> ${
                              ichimoku.span_b?.toFixed(2) || "N/A"
                            }</div>
                        `;
      }
    }

    // Update SuperTrend
    if (indicators.supertrend) {
      const supertrend = indicators.supertrend;
      const supertrendElement = document.getElementById("supertrendData");
      if (supertrendElement) {
        supertrendElement.innerHTML = `
                            <div><strong>Value:</strong> ${
                              supertrend.value?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>Direction:</strong> ${
                              supertrend.direction === 1
                                ? "Uptrend"
                                : "Downtrend"
                            }</div>
                            <div><strong>Signal:</strong> ${
                              supertrend.signal || "N/A"
                            }</div>
                            <div><strong>ATR:</strong> ${
                              supertrend.atr?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>Upper Band:</strong> ${
                              supertrend.upper_band?.toFixed(2) || "N/A"
                            }</div>
                            <div><strong>Lower Band:</strong> ${
                              supertrend.lower_band?.toFixed(2) || "N/A"
                            }</div>
                        `;
      }
    }

    if (indicators.stochastic) {
      const stochastic = indicators.stochastic;
      const stochasticElement = document.getElementById("stochasticData");
      console.log("Stochastic Element : ", stochasticElement);

      if (stochasticElement) {
        const kValue = stochastic.k || stochastic.k_percent || stochastic["%k"];
        const dValue = stochastic.d || stochastic.d_percent || stochastic["%d"];
        const signal = stochastic.signal || "N/A";

        // Generate zone information if not provided
        let zone = stochastic.zone;
        if (!zone && kValue !== undefined) {
          if (kValue > 80) {
            zone = "Overbought (>80)";
          } else if (kValue < 20) {
            zone = "Oversold (<20)";
          } else {
            zone = "Normal (20-80)";
          }
        }

        stochasticElement.innerHTML = `
                            <div><strong>%K:</strong> ${
                              kValue !== undefined ? kValue.toFixed(2) : "N/A"
                            }</div>
                            <div><strong>%D:</strong> ${
                              dValue !== undefined ? dValue.toFixed(2) : "N/A"
                            }</div>
                            <div><strong>Signal:</strong> ${signal}</div>
                            <div><strong>Zone:</strong> ${zone || "N/A"}</div>
                        `;
      }
    }

    // Update CCI
    if (indicators.cci) {
      const cci = indicators.cci;
      const cciElement = document.getElementById("cciData");
      if (cciElement) {
        const cciValue = cci.value || 0;
        const signal = cci.signal || "N/A";
        // Generate zone information
        let zone;
        if (cciValue > 100) {
          zone = "Overbought (>100)";
        } else if (cciValue < -100) {
          zone = "Oversold (<-100)";
        } else if (cciValue > 0) {
          zone = "Bullish (0-100)";
        } else {
          zone = "Bearish (-100-0)";
        }

        cciElement.innerHTML = `
                            <div><strong>CCI:</strong> ${cciValue.toFixed(
                              2
                            )}</div>
                            <div><strong>Signal:</strong> ${signal}</div>
                            <div><strong>Zone:</strong> ${zone}</div>
                        `;
      }
    }

    if (indicators.roc) {
      const roc = indicators.roc;
      const rocElement = document.getElementById("rocData");
      if (rocElement) {
        rocElement.innerHTML = `
                            <div><strong>ROC:</strong> ${
                              roc.value?.toFixed(2) || "N/A"
                            }%</div>
                            <div><strong>Signal:</strong> <span>${
                              roc.signal || "N/A"
                            }</span></div>
                        `;
      }
    }

    // Update momentum
    if (indicators.momentum) {
      const momentum = indicators.momentum;
      const momentumElement = document.getElementById("momentumData");
      if (momentumElement) {
        const value = momentum.value || 0;
        const signal = momentum.signal || "N/A";

        // Color coding based on signal
        let signalColor = "#333";
        if (signal.includes("STRONG_BULLISH")) signalColor = "#27ae60";
        else if (signal.includes("BULLISH")) signalColor = "#2ecc71";
        else if (signal.includes("STRONG_BEARISH")) signalColor = "#e74c3c";
        else if (signal.includes("BEARISH")) signalColor = "#c0392b";
        else if (signal === "NEUTRAL") signalColor = "#95a5a6";

        momentumElement.innerHTML = `
                            <div><strong>Value:</strong> ${value.toFixed(
                              4
                            )}</div>
                            <div><strong>Signal:</strong> <span style="color: ${signalColor}; font-weight: bold;">${signal}</span></div>
                        `;
      }
    }

    // ATR
    if (indicators.atr) {
      const atr = indicators.atr;
      const atrElement = document.getElementById("atrData");
      if (atrElement) {
        const value = atr.value || 0;
        const signal = atr.signal || "N/A";

        // Parse complex ATR signals for better display
        let signalDisplay = signal;
        let signalColor = "#333";

        if (signal.includes("HIGH_VOLATILITY")) {
          signalColor = "#e67e22";
        } else if (signal.includes("LOW_VOLATILITY")) {
          signalColor = "#3498db";
        } else if (signal.includes("INCREASING")) {
          signalColor = "#e74c3c";
        } else if (signal.includes("DECREASING")) {
          signalColor = "#27ae60";
        } else if (signal.includes("STABLE")) {
          signalColor = "#95a5a6";
        }

        // Handle complex signals with multiple components
        if (signal.includes("|")) {
          const signalParts = signal.split(" | ");
          signalDisplay = `<div style="color: ${signalColor}; font-size: 0.9em;">
                                ${signalParts
                                  .map(
                                    (part) =>
                                      `<div>• ${part.replace(/_/g, " ")}</div>`
                                  )
                                  .join("")}
                            </div>`;
        } else {
          signalDisplay = `<span style="color: ${signalColor}; font-weight: bold;">${signal.replace(
            /_/g,
            " "
          )}</span>`;
        }

        atrElement.innerHTML = `
                            <div><strong>ATR Value:</strong> ${value.toFixed(
                              4
                            )}</div>
                            <div><strong>Volatility Signal:</strong> ${signalDisplay}</div>
                        `;
      }
    }

    // Update Fibonacci
    if (indicators.fibonacci) {
      const fibonacci = indicators.fibonacci;
      const fibonacciElement = document.getElementById("fibonacciData");

      if (fibonacciElement) {
        let fibHtml = `
                            <div><strong>Range:</strong> ₹${
                              fibonacci.swing_high?.toFixed(2) || "N/A"
                            } - ₹${
          fibonacci.swing_low?.toFixed(2) || "N/A"
        }</div>
                            <div><strong>Signal:</strong> ${
                              fibonacci.signal || "N/A"
                            }</div>
                        `;

        if (fibonacci.levels) {
          fibHtml +=
            '<div style="margin-top: 8px; font-size: 0.9em;"><strong>Levels:</strong></div>';

          const fibLevels = [
            { key: "fib_0.236", percentage: "23.6%" },
            { key: "fib_0.382", percentage: "38.2%" },
            { key: "fib_0.500", percentage: "50.0%" },
            { key: "fib_0.618", percentage: "61.8%" },
            { key: "fib_0.786", percentage: "78.6%" },
          ];

          for (const fibLevel of fibLevels) {
            const value = fibonacci.levels[fibLevel.key];
            if (value !== null && value !== undefined && !isNaN(value)) {
              fibHtml += `<div style="font-size: 0.85em;">• ${
                fibLevel.percentage
              }: ₹${value.toFixed(2)}</div>`;
            }
          }
        }

        fibonacciElement.innerHTML = fibHtml;
      }
    }
  } catch (error) {
    console.error("Error updating indicators display:", error);
    addLog("❌ Error updating indicators display: " + error.message, "error");
  }
}

// WebSocket connection functions
function connectWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    addLog("Already connected", "info");
    return;
  }

  try {
    addLog("Connecting to WebSocket...", "info");

    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;

    addLog(`WebSocket URL: ${wsUrl}`, "info");
    console.log("Attempting WebSocket connection to:", wsUrl);

    ws = new WebSocket(wsUrl);

    ws.onopen = function (event) {
      addLog("✅ WebSocket connected successfully!", "success");
      updateConnectionStatus(true);
      connectionStartTime = Date.now();
      startConnectionTimer();
      updateButtons(true);
    };

    ws.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        messageCount++;
        updateTradingData(data);
        updatePriceHistory(data.price);

        // Fetch technical indicators from API
        fetchTechnicalIndicators();

        updateStats();
        addLog(
          `📊 Received: ${data.ticker} @ ₹${Number(data.price).toFixed(
            2
          )} (V: ${Number(data.volume).toLocaleString()})`,
          "info"
        );
      } catch (error) {
        addLog(`❌ Error parsing message: ${error.message}`, "error");
      }
    };

    ws.onclose = function (event) {
      addLog(
        `🔌 WebSocket disconnected (Code: ${event.code}, Reason: ${
          event.reason || "No reason"
        })`,
        "error"
      );
      updateConnectionStatus(false);
      stopConnectionTimer();
      updateButtons(false);
      ws = null;
    };

    ws.onerror = function (error) {
      addLog(`❌ WebSocket error occurred`, "error");
      console.error("WebSocket error details:", error);
      updateConnectionStatus(false);
      updateButtons(false);
    };
  } catch (error) {
    addLog(`❌ Error creating WebSocket connection: ${error.message}`, "error");
    updateConnectionStatus(false);
    updateButtons(false);
  }
}

function disconnectWebSocket() {
  try {
    if (ws) {
      ws.close(1000, "User initiated disconnect");
      ws = null;
    }
    addLog("🔌 Disconnected by user", "info");
  } catch (error) {
    addLog(`❌ Error disconnecting WebSocket: ${error.message}`, "error");
  }
}

function updateConnectionStatus(connected) {
  try {
    const indicator = document.getElementById("statusIndicator");
    const statusText = document.getElementById("statusText");

    if (indicator && statusText) {
      if (connected) {
        indicator.classList.add("connected");
        statusText.textContent = "Connected";
      } else {
        indicator.classList.remove("connected");
        statusText.textContent = "Disconnected";
      }
    }
  } catch (error) {
    console.error("Error updating connection status:", error);
  }
}

function updateButtons(connected) {
  try {
    const connectBtn = document.getElementById("connectBtn");
    const disconnectBtn = document.getElementById("disconnectBtn");

    if (connectBtn && disconnectBtn) {
      connectBtn.disabled = connected;
      disconnectBtn.disabled = !connected;
    }
  } catch (error) {
    console.error("Error updating buttons:", error);
  }
}

function updateTradingData(data) {
  try {
    const elements = {
      ticker: data.ticker,
      price: `₹${Number(data.price).toFixed(2)}`,
      open: `₹${Number(data.open).toFixed(2)}`,
      high: `₹${Number(data.high).toFixed(2)}`,
      low: `₹${Number(data.low).toFixed(2)}`,
      close: `₹${Number(data.close).toFixed(2)}`,
      volume: Number(data.volume).toLocaleString(),
      timestamp: new Date(data.timestamp).toLocaleString(),
    };

    for (const [id, value] of Object.entries(elements)) {
      const element = document.getElementById(id);
      if (element) {
        element.textContent = value;
      }
    }
  } catch (error) {
    addLog(`❌ Error updating trading data: ${error.message}`, "error");
  }
}

function updatePriceHistory(price) {
  try {
    priceHistory.push({
      price: Number(price),
      time: new Date(),
    });

    if (priceHistory.length > maxPriceHistory) {
      priceHistory.shift();
    }

    updateChart();
  } catch (error) {
    addLog(`❌ Error updating price history: ${error.message}`, "error");
  }
}

function updateChart() {
  try {
    const chartContainer = document.getElementById("chartContainer");
    if (!chartContainer) return;

    if (priceHistory.length === 0) {
      chartContainer.innerHTML = "<div>No data available</div>";
      return;
    }

    const latestPrice = priceHistory[priceHistory.length - 1].price;
    const minPrice = Math.min(...priceHistory.map((p) => p.price));
    const maxPrice = Math.max(...priceHistory.map((p) => p.price));
    const priceChange =
      priceHistory.length > 1 ? latestPrice - priceHistory[0].price : 0;
    const priceChangePercent =
      priceHistory.length > 1 ? (priceChange / priceHistory[0].price) * 100 : 0;

    chartContainer.innerHTML = `
                    <div style="text-align: center; width: 100%;">
                        <div style="font-size: 2em; font-weight: bold; color: ${
                          priceChange >= 0 ? "#27ae60" : "#e74c3c"
                        };">
                            ₹${latestPrice.toFixed(2)}
                        </div>
                        <div style="font-size: 1.2em; color: ${
                          priceChange >= 0 ? "#27ae60" : "#e74c3c"
                        }; margin: 10px 0;">
                            ${priceChange >= 0 ? "+" : ""}${priceChange.toFixed(
      2
    )} (${priceChangePercent.toFixed(2)}%)
                        </div>
                        <div style="font-size: 0.9em; color: #7f8c8d;">
                            Range: ₹${minPrice.toFixed(
                              2
                            )} - ₹${maxPrice.toFixed(2)}
                        </div>
                        <div style="font-size: 0.9em; color: #7f8c8d; margin-top: 5px;">
                            ${priceHistory.length} data points
                        </div>
                    </div>
                `;
  } catch (error) {
    console.error("Error updating chart:", error);
  }
}

function updateStats() {
  try {
    const messageCountElement = document.getElementById("messageCount");
    if (messageCountElement) {
      messageCountElement.textContent = messageCount;
    }
  } catch (error) {
    console.error("Error updating stats:", error);
  }
}

function startConnectionTimer() {
  try {
    stopConnectionTimer(); // Clear any existing timer
    connectionTimer = setInterval(() => {
      if (connectionStartTime) {
        const elapsed = Math.floor((Date.now() - connectionStartTime) / 1000);
        const connectionTimeElement = document.getElementById("connectionTime");
        if (connectionTimeElement) {
          connectionTimeElement.textContent = `${elapsed}s`;
        }
      }
    }, 1000);
  } catch (error) {
    console.error("Error starting connection timer:", error);
  }
}

function stopConnectionTimer() {
  if (connectionTimer) {
    clearInterval(connectionTimer);
    connectionTimer = null;
  }
}

function addLog(message, type = "info") {
  try {
    const logs = document.getElementById("logs");
    if (!logs) return;

    const logEntry = document.createElement("div");
    logEntry.className = "log-entry";

    const time = new Date().toLocaleTimeString();
    logEntry.innerHTML = `
                    <span class="log-time">[${time}]</span>
                    <span class="log-message ${type}">${message}</span>
                `;

    logs.appendChild(logEntry);
    logs.scrollTop = logs.scrollHeight;

    // Limit log entries to prevent memory issues
    while (logs.children.length > 1000) {
      logs.removeChild(logs.firstChild);
    }
  } catch (error) {
    console.error("Error adding log:", error);
  }
}

function clearLogs() {
  try {
    const logs = document.getElementById("logs");
    if (logs) {
      logs.innerHTML = "";
    }
    messageCount = 0;
    updateStats();
    addLog("🧹 Logs cleared", "info");
  } catch (error) {
    console.error("Error clearing logs:", error);
  }
}

// Initialize the application
function initializeApp() {
  try {
    addLog("🚀 Trading Dashboard loaded", "info");
    addLog('Click "Connect" to start receiving live data', "info");

    // Auto-connect after 2 seconds if not in development mode
    if (
      window.location.hostname !== "localhost" &&
      window.location.hostname !== "127.0.0.1"
    ) {
      setTimeout(() => {
        addLog("🔄 Auto-connecting to WebSocket...", "info");
        connectWebSocket();
      }, 2000);
    }

    // Start periodic fetch of technical indicators
    setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        fetchTechnicalIndicators();
      }
    }, 10000); // Fetch every 10 seconds to reduce API load
  } catch (error) {
    addLog(`❌ Error initializing app: ${error.message}`, "error");
  }
}

// Event listeners
window.addEventListener("load", initializeApp);

window.addEventListener("beforeunload", function () {
  try {
    if (ws) {
      ws.close();
    }
  } catch (error) {
    console.error("Error during page unload:", error);
  }
});

// Handle visibility change to pause/resume when tab is not active
document.addEventListener("visibilitychange", function () {
  if (document.hidden) {
    // Tab is now hidden - could pause some operations
    console.log("Tab hidden - reducing activity");
  } else {
    // Tab is now visible - resume full activity
    console.log("Tab visible - resuming full activity");
    if (ws && ws.readyState === WebSocket.OPEN) {
      fetchTechnicalIndicators();
    }
  }
});
